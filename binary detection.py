import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.signal import get_window, find_peaks
import queue

# ===================== USER PARAMETERS =====================
Fs = 44100
WINDOW_SIZE = 4096
HOP_SIZE = 2048

# Algorithm Parameters
R_HZ = 75.0
min_dist = 21
SIGMA_THRESH = 1.5

T_SECONDS = 2.0
T_FRAMES = int(T_SECONDS * (Fs / HOP_SIZE))
T_FRAMES_LOW_BOUND = int((T_SECONDS - 0.5 * T_SECONDS) * (Fs / HOP_SIZE))

# Visualization
VMIN, VMAX = -100, 50

# ===================== INTERNAL SETUP =====================
freqs = np.fft.rfftfreq(WINDOW_SIZE, 1 / Fs)


def hz_to_idx(hz):
    return int(np.clip(hz * WINDOW_SIZE / Fs, 0, len(freqs) - 1))


R_BINS = hz_to_idx(R_HZ)
min_dist_bins = hz_to_idx(min_dist)

audio_q = queue.Queue()
window_func = get_window('hann', WINDOW_SIZE)
input_buffer = np.zeros(WINDOW_SIZE - HOP_SIZE)

# Dictionary Structure:
# { frequency_hz: { 'count': int, 'flag': bool } }
persistent_tracks = {}


# ===================== AUDIO CALLBACK =====================
def callback(indata, frames, time, status):
    if status: print(status)
    audio_q.put(indata[:, 0].copy())


# ===================== MAIN LOOP =====================
def update(frame):
    global input_buffer, persistent_tracks

    try:
        new_data = audio_q.get_nowait()
    except queue.Empty:
        return line, tracker_points, avg_points, connector_lines

    # Stitch buffer
    full_frame = np.concatenate((input_buffer, new_data))
    input_buffer = full_frame[-len(input_buffer):]

    # STFT Step
    frame_windowed = full_frame * window_func
    fft_mag = np.abs(np.fft.rfft(frame_windowed))
    spec_db = 20 * np.log10(fft_mag + 1e-12)

    # --- Step 5: Check Persistence (Existing Tracks) ---
    keys_to_remove = []

    # Visual Lists to populate
    vis_freqs = []  # X coordinates
    vis_peaks = []  # Y coordinates (Peak)
    vis_avgs = []  # Y coordinates (Average)

    # For connecting lines, we need a flat list with NaNs to break the lines
    # Format: [x1, x1, nan, x2, x2, nan...]
    conn_x = []
    conn_y = []

    # Iterate over existing keys. NOTE: We cannot delete keys while iterating,
    # so we add them to 'keys_to_remove' and delete later.
    for center_freq, data in persistent_tracks.items():
        # 1. Define window around the TRACKED frequency
        idx_center = hz_to_idx(center_freq)
        idx_start = max(0, idx_center - R_BINS)
        idx_end = min(len(spec_db), idx_center + R_BINS)

        # Define smaller "nearby" window for the peak check
        idx_nearby_start = max(0, idx_center - min_dist_bins)
        idx_nearby_end = min(len(spec_db), idx_center + min_dist_bins)

        # 2. Freshly calculate stats for THIS frame
        local_region = spec_db[idx_start:idx_end]
        if len(local_region) == 0: continue

        curr_avg = np.mean(local_region)
        curr_sigma = np.std(local_region)

        # --- VISUALIZATION DATA CAPTURE START ---
        # Capture data for this track to visualize it
        current_db = spec_db[idx_center]
        vis_freqs.append(center_freq)
        vis_peaks.append(current_db)
        vis_avgs.append(curr_avg)

        # Build the vertical connector line (Avg -> Peak)
        conn_x.extend([center_freq, center_freq, np.nan])
        conn_y.extend([curr_avg, current_db, np.nan])
        # --- VISUALIZATION DATA CAPTURE END ---

        # 3. Dynamic Threshold check
        threshold = curr_avg + SIGMA_THRESH * curr_sigma

        nearby_region = spec_db[idx_nearby_start:idx_nearby_end]
        if len(nearby_region) == 0: continue
        window_max = np.max(nearby_region)

        if window_max >= threshold:
            # Increment logic: Clamp at T_FRAMES
            data['count'] = min(T_FRAMES, data['count'] + 1)

            # Set flag if we reached the target confidence
            if data['count'] == T_FRAMES:
                data['flag'] = True
        else:
            # Decrement logic (Grace period)
            data['count'] -= 1

            # If it drops below the low bound, mark for deletion
            if data['count'] <= T_FRAMES_LOW_BOUND:
                keys_to_remove.append(center_freq)

    # Clean up lost tracks
    for k in keys_to_remove:
        del persistent_tracks[k]

    # --- Step 2: Find Local Maxima ---
    peak_indices, _ = find_peaks(spec_db, distance=R_BINS, height=VMIN)

    # --- Steps 3 & 4: Filter New Candidates ---
    for p_idx in peak_indices:
        p_freq = freqs[p_idx]
        p_val = spec_db[p_idx]

        s_start = max(0, p_idx - R_BINS)
        s_end = min(len(spec_db), p_idx + R_BINS)
        local_region = spec_db[s_start:s_end]

        local_avg = np.mean(local_region)
        local_sigma = np.std(local_region)

        # Check Condition: Value > Mean + SIGMA_THRESH*Sigma
        if p_val > (local_avg + (SIGMA_THRESH * local_sigma)):

            # Check overlap with existing tracks
            is_new = True
            for existing_f in persistent_tracks:
                if abs(existing_f - p_freq) < min_dist:
                    is_new = False
                    break

            if is_new:
                # Initialize New Track
                persistent_tracks[p_freq] = {
                    'count': 0,
                    'flag': False
                }

    # --- Step 6: Reporting ---
    detected_msg = []
    for f, data in persistent_tracks.items():
        # Only report if the flag is True (confirmed persistent)
        if data['flag']:
            detected_msg.append(f"{f:.1f}Hz")

    if detected_msg:
        print(f"Detected: {', '.join(detected_msg)}")

    # Update Plot Data
    line.set_ydata(spec_db)

    # Update the dots, averages, and connector lines
    tracker_points.set_data(vis_freqs, vis_peaks)
    avg_points.set_data(vis_freqs, vis_avgs)
    connector_lines.set_data(conn_x, conn_y)

    return line, tracker_points, avg_points, connector_lines


# ===================== PLOT SETUP =====================
fig, ax = plt.subplots()
ax.set_ylim(VMIN, VMAX)
ax.set_xlim(0, 5000)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude (dB)")
ax.set_title(f"Dynamic Threshold Detector (SIGMA_THRESH={SIGMA_THRESH})")

# The main spectrum line
line, = ax.plot(freqs, np.zeros_like(freqs), label='Spectrum')

# 1. Vertical Grey lines connecting Average to Peak (visualizes "how much above")
connector_lines, = ax.plot([], [], color='gray', linewidth=1, alpha=0.7)

# 2. Green horizontal markers for the Local Average
avg_points, = ax.plot([], [], 'g_', markersize=10, markeredgewidth=2, label='Local Avg')

# 3. Red Dots for the Persistent Frequency Peaks
tracker_points, = ax.plot([], [], 'ro', markersize=6, label='Tracked Peak')

ax.legend(loc='upper right')

stream = sd.InputStream(samplerate=Fs, blocksize=HOP_SIZE, channels=1, callback=callback)

with stream:
    ani = animation.FuncAnimation(fig, update, interval=10, blit=True, cache_frame_data=False)
    plt.show()