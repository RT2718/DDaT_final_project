import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, sosfilt, sosfilt_zi, find_peaks
import queue
import serial  # ### --- SERIAL ADDITION --- ###

# ==========================================
# 1. Global Parameters & Pre-computations
# ==========================================
FS = 44100
C = 343.0
D = 0.049497
BLOCKSIZE = 4096 * 4
FFT_SIZE = BLOCKSIZE

# We now control filtering dynamically, so static freq bounds are less critical,
# but we keep them as an absolute floor/ceiling if desired.
FREQ_L = 300.0
FREQ_H = 2000.0

freq_filtering = False  # kept this at False, used butterworth filtering instead
whitening = False
lawson = False
butter_filter = True

# --- Dynamic Filter Globals ---
WINDOW_HZ = 300.0  # How wide the bandpass should be around the target
current_filter_target = -1.0
current_sos = None
filter_state = None

r = D / np.sqrt(2)
EMA_ALPHA = 0.4
ema_vector = None

r_mn = np.array([
    [-D / 2, -D / 2, 0],
    [-D / 2, D / 2, 0],
    [D / 2, D / 2, 0],
    [D / 2, -D / 2, 0]
])

lags = (np.arange(FFT_SIZE) - FFT_SIZE // 2) / FS
phi_mn = [(5 / 4) * np.pi, (3 / 4) * np.pi, (1 / 4) * np.pi, (7 / 4) * np.pi]


def build_pairs(N):
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


pairs = build_pairs(len(r_mn))

A_matrix = np.zeros((len(pairs), 2))
tau_max_array = np.zeros(len(pairs))

for k, (i, j) in enumerate(pairs):
    diff = (r_mn[j] - r_mn[i])
    A_matrix[k, :] = diff[:2]
    if (i, j) in [(1, 3), (3, 1), (0, 2), (2, 0)]:
        tau_max_array[k] = np.sqrt(2) * D / C
    else:
        tau_max_array[k] = D / C

window = np.hanning(BLOCKSIZE).astype(np.float32)
audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status: print(f"Audio Status: {status}")
    audio_queue.put(indata.copy())


# ==========================================
# 2. Binary Frequency Detection Engine
# ==========================================
R_HZ = 500.0
min_dist = 10
SIGMA_THRESH = 0.2
T_SECONDS = 1.5
T_FRAMES = max(1, int(T_SECONDS * (FS / BLOCKSIZE)))
T_FRAMES_LOW_BOUND = int(T_FRAMES * 0.5)

freqs_stft = np.fft.rfftfreq(BLOCKSIZE, 1 / FS)


def hz_to_idx(hz):
    return int(np.clip(hz * BLOCKSIZE / FS, 0, len(freqs_stft) - 1))


R_BINS = hz_to_idx(R_HZ)
min_dist_bins = hz_to_idx(min_dist)
persistent_tracks = {}


def update_frequency_tracker(channel_data):
    """
    Runs the STFT thresholding.
    RETURNS: A list of float frequencies currently flagged as persistent.
    """
    global persistent_tracks

    frame_windowed = channel_data * window
    fft_mag = np.abs(np.fft.rfft(frame_windowed))
    spec_db = 20 * np.log10(fft_mag + 1e-12)

    keys_to_remove = []

    for center_freq, data in persistent_tracks.items():
        idx_center = hz_to_idx(center_freq)
        idx_start = max(0, idx_center - R_BINS)
        idx_end = min(len(spec_db), idx_center + R_BINS)

        idx_nearby_start = max(0, idx_center - min_dist_bins)
        idx_nearby_end = min(len(spec_db), idx_center + min_dist_bins)

        local_region = spec_db[idx_start:idx_end]
        if len(local_region) == 0: continue

        curr_avg = np.mean(local_region)
        curr_sigma = np.std(local_region)
        threshold = curr_avg + SIGMA_THRESH * curr_sigma

        nearby_region = spec_db[idx_nearby_start:idx_nearby_end]
        if len(nearby_region) == 0: continue
        window_max = np.max(nearby_region)

        if window_max >= threshold:
            data['count'] = min(T_FRAMES, data['count'] + 1)
            if data['count'] == T_FRAMES:
                data['flag'] = True
        else:
            data['count'] -= 1
            if data['count'] <= T_FRAMES_LOW_BOUND:
                keys_to_remove.append(center_freq)

    for k in keys_to_remove:
        del persistent_tracks[k]

    peak_indices, _ = find_peaks(spec_db, distance=R_BINS, height=-100)
    for p_idx in peak_indices:
        p_freq = freqs_stft[p_idx]
        p_val = spec_db[p_idx]

        s_start = max(0, p_idx - R_BINS)
        s_end = min(len(spec_db), p_idx + R_BINS)
        local_region = spec_db[s_start:s_end]
        if len(local_region) == 0: continue

        local_avg = np.mean(local_region)
        local_sigma = np.std(local_region)

        if p_val > (local_avg + (SIGMA_THRESH * local_sigma)):
            is_new = True
            for existing_f in persistent_tracks:
                if abs(existing_f - p_freq) < min_dist:
                    is_new = False
                    break
            if is_new:
                # Ensure dictionary keys are native Python floats (hashable)
                persistent_tracks[float(p_freq)] = {'count': 0, 'flag': False, 'mag': p_val}

    # Gather active targets and return ONLY the one with highest magnitude
    active_targets = [(f, d['mag']) for f, d in persistent_tracks.items() if d['flag']]
    if active_targets:
        best_freq = max(active_targets, key=lambda x: x[1])[0]
        if FREQ_L <= best_freq <= FREQ_H:
            return best_freq
    else:
        return None


# ==========================================
# 3. Targeted TDOA and DOA Engine
# ==========================================
def calculate_delay_fourier(yi, yj, k, target_freq):
    Yi = np.fft.rfft(yi, n=FFT_SIZE)
    Yj = np.fft.rfft(yj, n=FFT_SIZE)

    freqs = np.fft.rfftfreq(FFT_SIZE, 1 / FS)

    # DYNAMIC SPECTRAL MASKING
    if freq_filtering:
        # Start with an all-zero mask
        freq_mask = np.zeros_like(freqs, dtype=bool)

        # Add windows around our targeted tracking frequencies
        # R_HZ defines how wide the bandpass filter is around the peak
        freq_mask &= (np.abs(freqs - target_freq) <= WINDOW_HZ)

        # Ensure we stay within absolute limits if desired
        freq_mask &= (freqs >= FREQ_L) & (freqs <= FREQ_H)

        # Apply the mask, zeroing out un-tracked noise
        Yi *= freq_mask
        Yj *= freq_mask

    G = Yi * np.conj(Yj)
    if np.all(np.abs(G) < 1e-12):
        return 0.0, 0.0, 0.0

    G_phat = G / (np.abs(G) + 1e-12)
    R = np.fft.irfft(G_phat, n=FFT_SIZE)
    R = np.fft.fftshift(R)

    max_tau = tau_max_array[k]
    valid = np.abs(lags) <= max_tau + 1 / FS
    valid_idx = np.flatnonzero(valid)
    lo_idx = int(valid_idx[0])
    hi_idx = int(valid_idx[-1])

    R_masked = np.where(valid, R, -np.inf)
    idx = int(np.argmax(R_masked))
    c1 = float(R_masked[idx])

    R_valid = R[lo_idx:hi_idx + 1]
    idx_main_rel = idx - lo_idx

    peaks = []

    # left edge
    if len(R_valid) >= 1:
        if len(R_valid) == 1 or R_valid[0] > R_valid[1]:
            peaks.append((0, R_valid[0]))

    # interior local maxima
    if len(R_valid) >= 3:
        is_peak = (R_valid[1:-1] > R_valid[:-2]) & (R_valid[1:-1] >= R_valid[2:])
        for rel_idx in np.flatnonzero(is_peak) + 1:
            peaks.append((rel_idx, R_valid[rel_idx]))

    # right edge
    if len(R_valid) >= 2:
        if R_valid[-1] > R_valid[-2]:
            peaks.append((len(R_valid) - 1, R_valid[-1]))

    # remove the main peak and near-neighbors if desired
    min_sep = 1
    cand = [(rel_idx, val) for rel_idx, val in peaks
            if abs(rel_idx - idx_main_rel) > min_sep]

    if cand:
        c2 = float(max(val for _, val in cand))
    else:
        c2 = 0.0

    # Parabolic sub-sample refinement
    idx_m1 = max(idx - 1, lo_idx)
    idx_p1 = min(idx + 1, hi_idx)
    Rm1, R0, Rp1 = R[idx_m1], R[idx], R[idx_p1]

    denom = Rm1 - 2.0 * R0 + Rp1
    delta = 0.5 * (Rm1 - Rp1) / denom if abs(denom) > 1e-12 else 0.0
    idx_refined = float(np.clip(idx + delta, lo_idx, hi_idx))

    tau = (idx_refined - FFT_SIZE // 2) / FS

    if lawson:
        P_LAWSON = 1.7
        LAWSON_SEARCH_RES = 30

        # Create local search grid around the current estimate
        shifts = np.linspace(-0.1 * max_tau, 0.1 * max_tau, LAWSON_SEARCH_RES)
        local_tau_grid = np.clip(tau + shifts, -max_tau, max_tau)

        # Frequency-domain phase shifts: shape (LAWSON_SEARCH_RES, Freq_bins)
        phase_shift = np.exp(-2j * np.pi * freqs[None, :] * local_tau_grid[:, None])

        # Apply phase shift to Yj and compute differences
        Yj_shifted = Yj[None, :] * phase_shift
        Y_diff = Yi[None, :] - Yj_shifted

        # Convert to time domain and compute Lp norm
        y_diff = np.fft.irfft(Y_diff, n=FFT_SIZE, axis=-1)
        Lp_norm = np.sum(np.abs(y_diff) ** P_LAWSON, axis=-1)  # shape (LAWSON_SEARCH_RES,)

        # Find the tau that minimizes the Lp norm
        best_idx = np.argmin(Lp_norm)
        tau = local_tau_grid[best_idx]
    return tau, c1, c2


def compute_doa(block, target_freq):
    global current_sos, current_filter_target, filter_state

    x = np.zeros((4, BLOCKSIZE), dtype=np.float32)
    x[0, :] = block[:, 4]
    x[1, :] = block[:, 3]
    x[2, :] = block[:, 2]
    x[3, :] = block[:, 1]

    if butter_filter:
        if current_sos is None or abs(target_freq - current_filter_target) > 0.0:
            current_filter_target = target_freq

            # Calculate bounds, ensuring we don't go below 20Hz or above Nyquist
            f_low = max(20.0, target_freq - WINDOW_HZ)
            f_high = min(FS / 2 - 20.0, target_freq + WINDOW_HZ)

            if f_low >= f_high:  # Fallback safety
                f_low, f_high = 20.0, FS / 2 - 20.0

            current_sos = butter(4, [f_low, f_high], btype='band', fs=FS, output='sos')
            filter_state = np.tile(sosfilt_zi(current_sos), (4, 1, 1))

        # Apply the continuous time-domain filter
        for ch in range(4):
            x[ch], filter_state[ch] = sosfilt(current_sos, x[ch], zi=filter_state[ch])

    x = x * window

    if whitening:
        R_x = np.cov(x) + np.eye(4) * 1e-6
        eigvals, eigvecs = np.linalg.eigh(R_x)
        eigvals = np.maximum(eigvals, 1e-10)
        W = np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        x = W @ x

    tau_measured = np.zeros(len(pairs))
    c1_measured = np.zeros(len(pairs))

    for k, (i, j) in enumerate(pairs):
        # Pass the targets down into the fourier calculation
        t, c1, _ = calculate_delay_fourier(x[i], x[j], k, target_freq)
        tau_measured[k] = t
        c1_measured[k] = c1

    # Simplify weights to use c1 magnitude since we bypassed c2
    weights = np.clip(c1_measured, 0, None)
    W = np.diag(weights)
    d = tau_measured * C

    At_W_A = A_matrix.T @ W @ A_matrix
    At_W_d = A_matrix.T @ W @ d
    u_xy = np.linalg.inv(At_W_A + 1e-10 * np.eye(2)) @ At_W_d

    mag_u_xy = np.linalg.norm(u_xy)
    mag_u_xy = np.clip(mag_u_xy, 0.0, 1.0)

    theta_raw = float(np.arcsin(mag_u_xy))
    phi_raw = float(np.arctan2(u_xy[1], u_xy[0])) % (2 * np.pi)

    return phi_raw, theta_raw


# ==========================================
# 4. Matplotlib 3D Polar Animation
# ==========================================
plt.ion()
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='polar')
ax.set_ylim(0, np.pi / 2)
ax.set_yticks([0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2])
ax.set_yticklabels(['$0$', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
ax.set_title("Targeted GCC-PHAT DOA")
peak_marker, = ax.plot([], [], 'ro', markersize=14, markeredgecolor='white', zorder=5)

# ### --- SERIAL GLOBAL --- ###
stm_serial = None


def update_plot(frame):
    global ema_vector, stm_serial
    latest_active_block = None
    active_target = None

    while not audio_queue.empty():
        block = audio_queue.get_nowait()
        if block is None or block.shape[0] < BLOCKSIZE: continue
        active_target = update_frequency_tracker(block[:, 1])
        if active_target is not None:
            latest_active_block = block

    if latest_active_block is not None and active_target is not None:
        raw_phi, raw_theta = compute_doa(latest_active_block, active_target)

        u_x = np.sin(raw_theta) * np.cos(raw_phi)
        u_y = np.sin(raw_theta) * np.sin(raw_phi)
        u_z = np.cos(raw_theta)
        current_vector = np.array([u_x, u_y, u_z])

        if ema_vector is None:
            ema_vector = current_vector
        else:
            ema_vector = (EMA_ALPHA * current_vector) + ((1.0 - EMA_ALPHA) * ema_vector)

        ema_vector /= (np.linalg.norm(ema_vector) + 1e-12)

        best_theta = np.arccos(np.clip(ema_vector[2], -1.0, 1.0))
        best_phi = np.arctan2(ema_vector[1], ema_vector[0])
        if best_phi < 0: best_phi += 2 * np.pi

        peak_marker.set_data([best_phi], [best_theta])
        peak_marker.set_visible(True)

        print(f"Tracking [{active_target:.1f}Hz] | Az: {np.rad2deg(best_phi):.1f}°, El: {np.rad2deg(best_theta):.1f}°")

        # ### --- SERIAL ADDITION: Transmit Data to STM32 --- ###
        if stm_serial is not None and stm_serial.is_open:
            try:
                # Format: "P:<phi>,T:<theta>\n"
                msg = f"P:{best_phi:.4f},T:{best_theta:.4f}\n"
                stm_serial.write(msg.encode('ascii'))
            except Exception as e:
                print(f"Serial write error: {e}")

    else:
        peak_marker.set_visible(False)
        print("Listening for persistent frequencies... (DOA Paused)    ", end='\r')

    return peak_marker,


# ==========================================
# 5. Main Execution
# ==========================================
if __name__ == "__main__":
    # ### --- SERIAL ADDITION: Open Port --- ###
    # Change 'COM3' to whatever your setup uses (e.g., 'COM4', '/dev/ttyUSB0')
    try:
        stm_serial = serial.Serial('COM3', baudrate=115200, timeout=0)
        print("Successfully opened serial connection to STM32.")
    except Exception as e:
        print(f"Warning: Could not open STM32 serial port. Running without motor control.\nError: {e}")

    dev_idx = None
    for i, dev in enumerate(sd.query_devices()):
        if 'respeaker' in dev['name'].lower() and dev['max_input_channels'] >= 6:
            dev_idx = i
            break

    print("Starting Targeted GCC-PHAT DOA Stream...")
    stream = sd.InputStream(device=dev_idx, samplerate=FS, channels=6 if dev_idx is not None else 1,
                            blocksize=BLOCKSIZE, dtype='float32', callback=audio_callback)
    with stream:
        ani = FuncAnimation(fig, update_plot, interval=50, blit=True, cache_frame_data=False)
        plt.show(block=True)

    # ### --- SERIAL ADDITION: Clean exit --- ###
    if stm_serial is not None and stm_serial.is_open:
        stm_serial.close()
