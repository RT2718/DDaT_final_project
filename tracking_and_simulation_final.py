import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.signal import butter, sosfilt, sosfilt_zi, find_peaks
import queue
import time

# Hardware libraries are optional: they are only needed for the LIVE microphone /
# motor path. Guarding the imports lets the SIMULATION run on any machine that
# does not have PortAudio (sounddevice) or a serial port available.
try:
    import sounddevice as sd
except Exception as _e:            # pragma: no cover - depends on host machine
    sd = None
    print(f"[info] sounddevice unavailable ({_e}); live audio disabled, simulation still works.")

try:
    import serial                  # ### --- SERIAL ADDITION --- ###
except Exception as _e:            # pragma: no cover - depends on host machine
    serial = None
    print(f"[info] pyserial unavailable ({_e}); motor output disabled.")

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

freq_filtering = False  # Force this to True so our mask is applied
whitening = False
lawson = False
butter_filter = True

# --- Dynamic Filter Globals ---
WINDOW_HZ = 300.0  # How wide the bandpass should be around the target
current_filter_target = -1.0
current_sos = None
filter_state = None

r = D / np.sqrt(2)
EMA_ALPHA = 0.8
ema_vector = None

r_mn = np.array([
    [-D / 2, -D / 2, 0],
    [-D / 2, D / 2, 0],
    [D / 2, D / 2, 0],
    [D / 2, -D / 2, 0]
])

lags = (np.arange(FFT_SIZE) - FFT_SIZE // 2) / FS
phi_mn = [(5 / 4) * np.pi, (3 / 4) * np.pi, (1 / 4) * np.pi, (7 / 4) * np.pi]

# ==========================================
# 1b. Simulation configuration
# ==========================================
# Set SIMULATE = True to run entirely in software (no microphone, no motor).
# The synthetic source emits a constant tone (so the frequency detector can lock
# onto it) plus broadband noise (so GCC-PHAT has phase content to work with), and
# is delayed per-microphone according to the REAL square-array geometry (r_mn).
SIMULATE = True
SIM_TRAJECTORY = 'circle'    # 'circle' (phi graph) | 'arc' (theta graph) | 'spiral' (both) | 'static'
SIM_TONE_FREQ = 1200.0    # Hz, must sit inside [FREQ_L, FREQ_H]
SIM_SOURCE_NOISE = 0.35   # broadband noise mixed into the source (helps GCC-PHAT)
SIM_SENSOR_NOISE = 0.02   # independent per-microphone noise (realism)

# Maps an algorithm microphone index (row of r_mn) to the hardware channel it is
# read from in compute_doa (x[0]=ch4, x[1]=ch3, x[2]=ch2, x[3]=ch1). The synthetic
# block is built with the SAME mapping so the recovered DOA matches the target.
CHANNEL_OF_MIC = [4, 3, 2, 1]


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
# 3b. Synthetic audio generator (simulation)
# ==========================================
def _direction_unit_vector(phi, theta):
    """theta = polar angle measured from zenith (0 = straight up, pi/2 = horizon),
    which is exactly the convention compute_doa recovers via arcsin(|u_xy|)."""
    return np.array([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta),
    ])


def generate_synthetic_block(phi, theta, tone_freq=SIM_TONE_FREQ,
                             source_noise=SIM_SOURCE_NOISE,
                             sensor_noise=SIM_SENSOR_NOISE):
    """Build one (BLOCKSIZE, 6) hardware-style block for a plane wave arriving from
    (phi, theta). Channels 1..4 carry the four raw mics (matching compute_doa's
    x[0]=ch4 ... x[3]=ch1); channels 0 and 5 are left as noise-only, unused stand-ins
    for the ReSpeaker's processed / playback-reference channels.

    The source is a constant tone (for the frequency detector) plus broadband noise
    (so GCC-PHAT has phase across the band). It is delayed per microphone in the
    frequency domain using the real array geometry r_mn.
    """
    n = BLOCKSIZE
    t = np.arange(n) / FS

    source = np.sin(2 * np.pi * tone_freq * t).astype(np.float32)
    source += source_noise * np.random.randn(n).astype(np.float32)

    S = np.fft.rfft(source)
    freqs = np.fft.rfftfreq(n, 1 / FS)

    u = _direction_unit_vector(phi, theta)

    block = np.zeros((n, 6), dtype=np.float32)
    for m in range(len(r_mn)):
        # A wavefront reaches a mic advanced by (r_m . u)/c relative to the origin,
        # i.e. a delay of tau_m = -(r_m . u)/c. Same sign convention as the small
        # circular-array simulator.
        tau_m = -np.dot(r_mn[m], u) / C
        Sm = S * np.exp(-1j * 2 * np.pi * freqs * tau_m)
        mic = np.fft.irfft(Sm, n=n).astype(np.float32)
        mic += sensor_noise * np.random.randn(n).astype(np.float32)
        block[:, CHANNEL_OF_MIC[m]] = mic

    # Fill the two unused channels with low-level noise so nothing downstream trips.
    block[:, 0] = sensor_noise * np.random.randn(n).astype(np.float32)
    block[:, 5] = sensor_noise * np.random.randn(n).astype(np.float32)
    return block


def _trajectory_target(trajectory, frame):
    """Return the (phi, theta_polar) target for a given trajectory and frame.
    Mirrors the small circular-array simulator, converted to the polar-from-zenith
    convention used here (theta_polar = pi/2 - elevation)."""
    t = frame
    if trajectory == 'circle':
        phi = (t / 50.0) * (2 * np.pi) % (2 * np.pi)
        theta = np.pi / 4  # constant, 45 deg from zenith
    elif trajectory == 'arc':
        phi = np.pi / 2
        elev = (np.sin(t / 20.0) + 1) / 2 * (np.pi / 2)
        theta = np.pi / 2 - elev
    elif trajectory == 'spiral':
        phi = (t / 20.0) * (2 * np.pi) % (2 * np.pi)
        elev = (np.sin(t / 50.0) + 1) / 2 * (np.pi / 2)
        theta = np.pi / 2 - elev
    else:  # static
        phi, theta = np.pi / 4, np.pi / 4
    # Keep theta away from the exact zenith where azimuth becomes ill-defined.
    theta = float(np.clip(theta, np.deg2rad(3.0), np.pi / 2))
    return phi, theta


# ==========================================
# 4. Matplotlib 3D Polar Animation  (LIVE mode only)
# ==========================================
fig = ax = peak_marker = None
stm_serial = None  # ### --- SERIAL GLOBAL --- ###

if not SIMULATE:
    plt.ion()
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection='polar')
    ax.set_ylim(0, np.pi / 2)
    ax.set_yticks([0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2])
    ax.set_yticklabels(['$0$', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
    ax.set_title("Targeted GCC-PHAT DOA")
    peak_marker, = ax.plot([], [], 'ro', markersize=14, markeredgecolor='white', zorder=5)


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
# 4b. Simulation runner
# ==========================================
def _setup_sim_plot(trajectory):
    """Error-vs-frame plot(s) mirroring the small circular-array simulator."""
    plt.ion()
    plt.rcParams.update({
        'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 16,
        'xtick.labelsize': 13, 'ytick.labelsize': 13,
    })
    fig_s = plt.figure(figsize=(10, 8))
    fig_s.suptitle(f"Detection + GCC-PHAT Tracking Error - Trajectory: {trajectory.capitalize()}",
                   fontsize=20, fontweight='bold')

    ax_phi = ax_theta = line_phi = line_theta = None
    if trajectory == 'circle':
        ax_phi = fig_s.add_subplot(1, 1, 1)
    elif trajectory == 'arc':
        ax_theta = fig_s.add_subplot(1, 1, 1)
    else:
        ax_phi = fig_s.add_subplot(2, 1, 1)
        ax_theta = fig_s.add_subplot(2, 1, 2)

    if ax_phi is not None:
        ax_phi.set_title('Error in Azimuth (Phi) [Detected - Target]')
        ax_phi.set_xlabel('Frame Number'); ax_phi.set_ylabel('Error [degrees]')
        ax_phi.grid(True)
        line_phi, = ax_phi.plot([], [], 'b-', linewidth=2.5)
    if ax_theta is not None:
        ax_theta.set_title('Error in Polar Angle (Theta) [Detected - Target]')
        ax_theta.set_xlabel('Frame Number'); ax_theta.set_ylabel('Error [degrees]')
        ax_theta.grid(True)
        line_theta, = ax_theta.plot([], [], 'r-', linewidth=2.5)

    fig_s.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig_s, ax_phi, ax_theta, line_phi, line_theta


def _print_sim_stats(ax, errors, name):
    if ax is None or not errors:
        return
    e = np.array(errors)
    stats = {'Mean': np.mean(e), 'MAE': np.mean(np.abs(e)),
             'RMSE': np.sqrt(np.mean(e ** 2)), 'Std Dev': np.std(e),
             'Max Abs': np.max(np.abs(e))}
    print(f"\n--- {name} ---")
    for k, v in stats.items():
        print(f"  {k + ':':<10} {v:>8.3f} deg")
    txt = f"RMSE: {stats['RMSE']:.2f}\u00b0\nMAE: {stats['MAE']:.2f}\u00b0\nMax: {stats['Max Abs']:.2f}\u00b0"
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.01, 0.95, txt, transform=ax.transAxes, va='top', bbox=props, fontsize=13)


def run_simulation(trajectory=SIM_TRAJECTORY, tone_freq=SIM_TONE_FREQ, show=True):
    """Software-only run: synthesize a moving tonal source, feed it through the REAL
    detection + GCC-PHAT DOA pipeline, and plot detected-vs-target angle error."""
    global persistent_tracks, ema_vector, current_sos, current_filter_target, filter_state

    # Reset all pipeline state for a clean run.
    persistent_tracks = {}
    ema_vector = None
    current_sos = None
    current_filter_target = -1.0
    filter_state = None

    if trajectory == 'circle':
        cycle_frames = 50
    elif trajectory == 'arc':
        cycle_frames = int(20 * 2 * np.pi)      # ~125 frames
    elif trajectory == 'spiral':
        cycle_frames = int(50 * 2 * np.pi)      # ~314 frames
    else:
        cycle_frames = 60

    # Warm-up frames (target held at its starting pose) let the frequency detector
    # lock and the EMA settle before we start scoring error.
    warmup = T_FRAMES + 3

    print("-" * 100)
    print(f"VIRTUAL SIMULATION | trajectory='{trajectory}' | tone={tone_freq:.0f} Hz | {cycle_frames} frames")
    print(f"(source: tone + broadband noise, delayed on the real square array; hardware not required)")
    print("-" * 100)

    fig_s, ax_phi, ax_theta, line_phi, line_theta = _setup_sim_plot(trajectory)

    frame_indices, phi_errors, theta_errors = [], [], []

    for raw_frame in range(-warmup, cycle_frames):
        frame = max(raw_frame, 0)
        tgt_phi, tgt_theta = _trajectory_target(trajectory, frame)

        block = generate_synthetic_block(tgt_phi, tgt_theta, tone_freq)

        active_target = update_frequency_tracker(block[:, 1])
        if active_target is None:
            continue  # not locked yet

        raw_phi, raw_theta = compute_doa(block, active_target)

        # Same EMA smoothing on the unit vector as the live path.
        cur = np.array([
            np.sin(raw_theta) * np.cos(raw_phi),
            np.sin(raw_theta) * np.sin(raw_phi),
            np.cos(raw_theta),
        ])
        ema_vector = cur if ema_vector is None else EMA_ALPHA * cur + (1 - EMA_ALPHA) * ema_vector
        ema_vector = ema_vector / (np.linalg.norm(ema_vector) + 1e-12)

        best_theta = float(np.arccos(np.clip(ema_vector[2], -1.0, 1.0)))
        best_phi = float(np.arctan2(ema_vector[1], ema_vector[0])) % (2 * np.pi)

        if raw_frame < 0:
            continue  # still warming up; don't score

        err_phi = (best_phi - tgt_phi + np.pi) % (2 * np.pi) - np.pi
        err_theta = best_theta - tgt_theta

        frame_indices.append(frame)
        phi_errors.append(np.rad2deg(err_phi))
        theta_errors.append(np.rad2deg(err_theta))

        if line_phi is not None:
            line_phi.set_data(frame_indices, phi_errors)
            ax_phi.relim(); ax_phi.autoscale_view()
        if line_theta is not None:
            line_theta.set_data(frame_indices, theta_errors)
            ax_theta.relim(); ax_theta.autoscale_view()

        if show:
            plt.draw(); plt.pause(0.001)

        print(f"frame {frame:4d} | tone {active_target:6.1f}Hz | "
              f"phi {np.rad2deg(best_phi):6.1f}\u00b0 (tgt {np.rad2deg(tgt_phi):6.1f}\u00b0) | "
              f"theta {np.rad2deg(best_theta):5.1f}\u00b0 (tgt {np.rad2deg(tgt_theta):5.1f}\u00b0)")

    print("\n" + "=" * 50)
    print("        DOA TRACKING ERROR STATISTICS")
    print("=" * 50)
    _print_sim_stats(ax_phi, phi_errors, "Azimuth (Phi)")
    _print_sim_stats(ax_theta, theta_errors, "Polar (Theta)")
    print("=" * 50)

    if show:
        plt.draw()
        print("\nSimulation complete. Close the graph window to exit.")
        plt.ioff()
        plt.show()

    return {'frames': frame_indices, 'phi_errors': phi_errors, 'theta_errors': theta_errors}


# ==========================================
# 5. Main Execution
# ==========================================
def run_live():
    global stm_serial
    # ### --- SERIAL ADDITION: Open Port --- ###
    # Change 'COM3' to whatever your setup uses (e.g., 'COM4', '/dev/ttyUSB0')
    if serial is not None:
        try:
            stm_serial = serial.Serial('COM3', baudrate=115200, timeout=0)
            print("Successfully opened serial connection to STM32.")
        except Exception as e:
            print(f"Warning: Could not open STM32 serial port. Running without motor control.\nError: {e}")
    else:
        print("Warning: pyserial not installed. Running without motor control.")

    if sd is None:
        raise RuntimeError("sounddevice is not available; cannot run live audio. "
                           "Set SIMULATE = True to run in software.")

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


if __name__ == "__main__":
    if SIMULATE:
        run_simulation(SIM_TRAJECTORY, SIM_TONE_FREQ)
    else:
        run_live()