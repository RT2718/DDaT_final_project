import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.optimize import minimize_scalar
import queue

# --- 1. Global Parameters & Pre-computations ---
FS = 16000
C = 343.0
D = 0.049497
BLOCKSIZE = 4096
FFT_SIZE = BLOCKSIZE
FREQ_L = 750.0  # Lower frequency bound in Hz
FREQ_H = 850.0  # Upper frequency bound in Hz (e.g., standard speech band)
freq_filtering = False  # optional optimizations
whitening = False
lawson = True
R = D / np.sqrt(2)

# --- EMA Filter Parameters ---
EMA_ALPHA = 0.2  # Smoothing factor (0.0 to 1.0). Lower = smoother but slower to react.
ema_vector = None  # Stores the current state of the filtered vector

# Microphone geometry (m=0 is the reference)
r_mn = np.array([
    [-D / 2, -D / 2, 0],  # m = 0
    [-D / 2, D / 2, 0],  # m = 1
    [D / 2, D / 2, 0],  # m = 2
    [D / 2, -D / 2, 0]  # m = 3
])

phi_mn = [(5 / 4) * np.pi, (3 / 4) * np.pi, (1 / 4) * np.pi, (7 / 4) * np.pi]


def build_pairs(N):
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


pairs = build_pairs(len(r_mn))

window = np.hanning(BLOCKSIZE).astype(np.float32)
audio_queue = queue.Queue()


def audio_callback(indata, frames, time, status):
    if status: print(f"Audio Status: {status}")
    audio_queue.put(indata.copy())


# --- 2. TDOA and DOA Engine ---

def calculate_delay_fourier(yi, yj, i, j):
    """
    Calculates sub-sample TDOA between two signals using GCC-PHAT.
    Convention: If j is closer to the source than i, tau_ij > 0
    """
    # FFT
    Yi = np.fft.rfft(yi, n=FFT_SIZE)
    Yj = np.fft.rfft(yj, n=FFT_SIZE)

    freqs = np.fft.rfftfreq(FFT_SIZE, 1 / FS)
    if freq_filtering:
        freq_mask = (freqs >= FREQ_L) & (freqs <= FREQ_H)
        Yi *= freq_mask
        Yj *= freq_mask

    # Cross-spectrum: X_i(w) * X_j^*(w)
    G = Yi * np.conj(Yj)
    if np.all(np.abs(G) < 1e-12):
        return 0.0

    G_phat = G / (np.abs(G) + 1e-12)

    # IFFT and shift
    R = np.fft.irfft(G_phat, n=FFT_SIZE)
    R = np.fft.fftshift(R)

    idx = int(np.argmax(R))

    # Parabolic sub-sample refinement
    idx_m1 = max(idx - 1, 0)
    idx_p1 = min(idx + 1, FFT_SIZE - 1)
    Rm1, R0, Rp1 = R[idx_m1], R[idx], R[idx_p1]

    denom = Rm1 - 2.0 * R0 + Rp1
    delta = 0.5 * (Rm1 - Rp1) / denom if abs(denom) > 1e-12 else 0.0
    idx_refined = idx + delta

    # Compute initial tau estimate from parabolic refinement
    tau = (idx_refined - FFT_SIZE // 2) / FS

    # Optional: Lawson norm optimization for refined TDOA
    if lawson:
        P_LAWSON = 1.7
        LAWSON_SEARCH_RES = 30
        # Determine max physical delay for this pair
        if (i, j) in [(1, 3), (3, 1), (0, 2), (2, 0)]:
            max_tau_pair = np.sqrt(2) * D / C
        else:
            max_tau_pair = D / C

        # Create local search grid around the current estimate
        shifts = np.linspace(- 0.1 * max_tau_pair, 0.1 * max_tau_pair, LAWSON_SEARCH_RES)
        local_tau_grid = tau + shifts

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

    return tau


def theoretical_tau(phi):
    """Calculates theoretical TDOA for all pairs given an azimuth phi."""
    u = np.array([np.cos(phi), np.sin(phi), 0.0])
    tau_theo = np.zeros(len(pairs))
    for k, (i, j) in enumerate(pairs):
        tau_theo[k] = R / C * (np.cos(phi - phi_mn[j]) - np.cos(phi - phi_mn[i]))  # or np.dot(r_mn[j] - r_mn[i], u) / C

    return tau_theo


def score_phi(phi, tau_measured):
    """
    Objective function to minimize: need to invert the sign for the minimize scalar algorithm
    """
    tau_theo = theoretical_tau(phi)
    return -float(np.dot(tau_theo, tau_measured))


def compute_doa(block):
    # Map physical mics (Assuming ReSpeaker raw mic indexing: channels 1-4)
    x = np.zeros((4, BLOCKSIZE), dtype=np.float32)
    x[0, :] = block[:, 4] * window  # m=0
    x[1, :] = block[:, 3] * window  # m=1
    x[2, :] = block[:, 2] * window  # m=2
    x[3, :] = block[:, 1] * window  # m=3

    if whitening:
        R_x = np.cov(x) + np.eye(4) * 1e-6  # Covariance with diagonal loading for stability
        eigvals, eigvecs = np.linalg.eigh(R_x)
        eigvals = np.maximum(eigvals, 1e-10)  # Prevent division by zero
        W = np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        x = W @ x  # Shape: (4, BLOCKSIZE)

    # 1. Calculate measured TDOA for all pairs
    tau_measured = np.zeros(len(pairs))
    for k, (i, j) in enumerate(pairs):
        tau_measured[k] = calculate_delay_fourier(x[i], x[j], i, j)

    # 2. Estimate Azimuth (Phi)
    # Coarse Grid Search
    grid = np.linspace(0.0, 2 * np.pi, 72, endpoint=False)
    scores = [score_phi(p, tau_measured) for p in grid]
    p0 = float(grid[int(np.argmin(scores))])

    # Fine Refinement
    half_step = np.pi / 72
    res = minimize_scalar(score_phi,
                          args=(tau_measured,),
                          bounds=(p0 - half_step, p0 + half_step),
                          method='bounded',
                          options={'xatol': 1e-7})
    phi_raw = float(res.x) % (2 * np.pi)

    # 3. Estimate Elevation (Theta) - Average over all pairs
    u_xy = np.array([np.cos(phi_raw), np.sin(phi_raw), 0.0])
    theta_ijs = []
    for k, (i, j) in enumerate(pairs):
        denom = R * (np.cos(phi_raw - phi_mn[j]) - np.cos(phi_raw - phi_mn[i]))  # np.dot(r_mn[j] - r_mn[i], u_xy)
        if abs(denom) > 1e-12:
            ratio = (tau_measured[k] * C) / denom
            ratio = np.clip(ratio, -1.0, 1.0)

            # theta_ij = pi/2 - arccos(...) as drawn
            theta_ij = np.pi / 2 - np.arccos(ratio)
            theta_ijs.append(theta_ij)

    if len(theta_ijs) > 0:
        theta_raw = float(np.mean(theta_ijs))
    else:
        theta_raw = np.pi / 2  # Fallback to horizon

    return phi_raw, theta_raw


# --- 3. Matplotlib 3D Polar Animation ---
plt.ion()
fig = plt.figure(figsize=(9, 8))
ax = fig.add_subplot(111, projection='polar')

# Set limits: Theta from 0 (zenith) to Pi/2 (horizon)
ax.set_ylim(0, np.pi / 2)
ax.set_yticks([0, np.pi / 8, np.pi / 4, 3 * np.pi / 8, np.pi / 2])
ax.set_yticklabels(['$0$', r'$\pi/8$', r'$\pi/4$', r'$3\pi/8$', r'$\pi/2$'])
ax.set_title("GCC-PHAT Pairwise 3D DOA\n(Radius = Theta, Angle = Phi)")

peak_marker, = ax.plot([], [], 'ro', markersize=14, markeredgecolor='white', zorder=5)


def update_plot(frame):
    global ema_vector

    block = None
    while not audio_queue.empty():
        block = audio_queue.get_nowait()

    if block is None or block.shape[0] < BLOCKSIZE:
        return peak_marker,

    raw_phi, raw_theta = compute_doa(block)

    # --- EMA FILTERING ---
    # Convert raw spherical angles to a 3D Cartesian unit vector
    u_x = np.sin(raw_theta) * np.cos(raw_phi)
    u_y = np.sin(raw_theta) * np.sin(raw_phi)
    u_z = np.cos(raw_theta)
    current_vector = np.array([u_x, u_y, u_z])

    # Apply the Exponential Moving Average to the vector
    if ema_vector is None:
        ema_vector = current_vector  # Initialize on first run
    else:
        ema_vector = (EMA_ALPHA * current_vector) + ((1.0 - EMA_ALPHA) * ema_vector)

    # Re-normalize the vector to keep it strictly on the unit sphere
    ema_vector /= (np.linalg.norm(ema_vector) + 1e-12)

    # Convert the filtered vector back to spherical angles
    best_theta = np.arccos(np.clip(ema_vector[2], -1.0, 1.0))
    best_phi = np.arctan2(ema_vector[1], ema_vector[0])
    if best_phi < 0:
        best_phi += 2 * np.pi  # Ensure phi stays in [0, 2pi]

    # Update plot marker
    peak_marker.set_data([best_phi], [best_theta])

    print(f"Raw Peak: {np.rad2deg(raw_phi):.1f}째 azimuth, {np.rad2deg(raw_theta):.1f}째 elev "
          f"| Smoothed Peak: {np.rad2deg(best_phi):.1f}째, {np.rad2deg(best_theta):.1f}째")

    return peak_marker,


# --- 4. Main Execution ---
if __name__ == "__main__":

    dev_idx = None
    for i, dev in enumerate(sd.query_devices()):
        if 'respeaker' in dev['name'].lower() and dev['max_input_channels'] >= 6:
            dev_idx = i
            break

    if dev_idx is None:
        print("Warning: ReSpeaker not found. Falling back to default input.")

    print(f"Starting Pairwise GCC-PHAT stream on device {dev_idx}. Close window to stop.")

    stream = sd.InputStream(device=dev_idx, samplerate=FS, channels=6 if dev_idx is not None else 1,
                            blocksize=BLOCKSIZE, dtype='float32', callback=audio_callback)

    with stream:
        ani = FuncAnimation(fig, update_plot, interval=50, blit=True, cache_frame_data=False)
        plt.show(block=True)