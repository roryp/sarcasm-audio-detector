import numpy as np
import subprocess, io
import soundfile as sf
import librosa, librosa.display
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle, Arc
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

"""
SARCASM AUDIO DETECTOR
======================
A tool for analyzing acoustic patterns in speech to detect sarcasm.
"""

file_path = 'voice.m4a'  # or the full path to your file

# 1) Figure out where the last 120 s start:
total_dur = librosa.get_duration(filename=file_path)
start_sec = max(0, total_dur - 120.0)

# 2) Use ffmpeg to grab that segment as a 16 kHz mono WAV in-memory:
cmd = [
    'ffmpeg',
    '-ss', str(start_sec),
    '-t', '120',
    '-i', file_path,
    '-ar', '16000',
    '-ac', '1',
    '-f', 'wav',
    'pipe:1'
]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
wav_bytes, _ = proc.communicate()

# 3) Read into numpy:
y, sr = sf.read(io.BytesIO(wav_bytes), dtype='float32')

# --- Spectrogram ---
n_fft = 1024
hop_spec = 512
D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_spec))
DB = librosa.amplitude_to_db(D, ref=np.max)

plt.figure(figsize=(10, 4))
plt.imshow(DB, origin='lower', aspect='auto', extent=[0, 120, 0, sr/2])
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('Spectrogram (Last 120 s)')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# --- Emotional Flow Scatter (4D) ---
frame_len = int(0.05 * sr)   # 50 ms frames
hop_len   = frame_len

rms      = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_len)[0]
times    = np.arange(len(rms)) * (hop_len / sr)

# Normalize to [0,1]
rms_n  = (rms - rms.min()) / (rms.max() - rms.min())
cent_n = (centroid - centroid.min()) / (centroid.max() - centroid.min())

sizes  = rms_n * 100 + 10    # marker size ~ arousal
colors = times              # color ~ time

plt.figure(figsize=(6, 6))
plt.scatter(rms_n, cent_n, s=sizes, c=colors)
plt.xlabel('Arousal Proxy (norm RMS)')
plt.ylabel('Valence Proxy (norm Spectral Centroid)')
plt.title('Emotional Flow Scatter (Last 120 s)')
cbar = plt.colorbar()
cbar.set_label('Time (s)')
plt.tight_layout()
plt.show()

# Create a single emotional flow scatter plot
plt.figure(figsize=(8, 8))
scatter = plt.scatter(rms_n, cent_n, s=sizes, c=colors, cmap='viridis')
plt.xlabel('Arousal Proxy (norm RMS)')
plt.ylabel('Valence Proxy (norm Spectral Centroid)')
plt.title('Emotional Flow Scatter (Last 120 s)')
cbar = plt.colorbar(scatter)
cbar.set_label('Time (s)')
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()

# Add an analysis plot to visualize emotional transitions
plt.figure(figsize=(10, 6))
# Find indices where there's significant change in either dimension
changes = []
threshold = 0.1
for i in range(1, len(rms_n)):
    # Calculate distance between consecutive points in 2D emotion space
    dist = np.sqrt((rms_n[i] - rms_n[i-1])**2 + (cent_n[i] - cent_n[i-1])**2)
    if dist > threshold:
        changes.append(i)

# Plot the emotional trajectory with highlighted transitions
plt.plot(times, rms_n, 'b-', label='Arousal', alpha=0.7)
plt.plot(times, cent_n, 'g-', label='Valence', alpha=0.7)

# Highlight areas of rapid emotional shifts
for c in changes:
    plt.axvline(x=times[c], color='r', linestyle='--', alpha=0.3)

# Add markers for potential sarcasm indicators (where arousal and valence diverge)
sarcasm_indicators = []
for i in range(len(rms_n)):
    # Look for points where arousal and valence have opposite values (one high, one low)
    if (rms_n[i] > 0.7 and cent_n[i] < 0.3) or (rms_n[i] < 0.3 and cent_n[i] > 0.7):
        sarcasm_indicators.append(i)

for s in sarcasm_indicators:
    plt.plot(times[s], rms_n[s], 'ro', markersize=8)

plt.xlabel('Time (s)')
plt.ylabel('Normalized Value')
plt.title('Emotional Transitions Analysis (Potential Sarcasm Indicators in Red)')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()