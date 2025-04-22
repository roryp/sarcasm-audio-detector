import numpy as np
import subprocess, io
import soundfile as sf
import librosa, librosa.display
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

"""
SLIDE NOTES: AUDIO ANALYSIS FOR SARCASM DETECTION

Slide 1: Audio Analysis Overview
- This analysis extracts acoustic features from speech that can reveal sarcasm
- Sarcasm relies on prosodic cues: pitch variations, timing, and emphasis patterns
- Our visualizations capture these subtle emotional markers that text analysis misses
- Using "The Void.mp3" as our case study for analysis

Slide 2: Spectrogram Visualization
- Spectrograms reveal frequency patterns over time, showing the "acoustic fingerprint" of speech
- Sarcastic speech shows distinctive energy distributions across frequencies
- Look for exaggerated emphasis patterns and unusual pitch contours
- Higher energy in certain frequencies can indicate emotional incongruence typical of sarcasm

Slide 3: Emotional Flow Visualization
- This 4D scatter plot tracks emotional dynamics that correlate with sarcastic speech:
  * X-axis (RMS): Represents vocal intensity/arousal - sarcasm often has distinctive emphasis patterns
  * Y-axis (Spectral Centroid): Represents tonal "brightness"/valence - sarcasm frequently shows valence mismatch 
  * Size: Indicates intensity of expression - larger points show emphasized segments
  * Color gradient: Represents time progression - reveals the temporal pattern of emotional shifts

Slide 4: Key Insights for Sarcasm Detection
- Sarcasm creates distinctive patterns in the emotional flow visualization:
  * Clusters showing high arousal but unexpected valence
  * Temporal transitions that differ from sincere speech
  * Exaggerated emphasis patterns visible as larger points
- These acoustic patterns allow detection of sarcasm even when text analysis would fail
- Next steps: Training ML models using these acoustic features to automatically detect sarcastic speech

Slide 5: Emotional Wheel Mapping
- The emotional flow visualization can be mapped to the emotion wheel model
- Sarcasm typically involves oscillation between contrasting emotions:
  * Movement between joy/ecstasy regions and disgust/loathing regions
  * Rapid transitions through anger, surprise and acceptance areas
  * These oscillation patterns are key indicators of emotional incongruence in sarcastic speech
- Each region of our 2D emotion space corresponds to emotion categories on the wheel:
  * Top-right: Joy, Ecstasy, Admiration (high arousal, positive valence)
  * Bottom-right: Serenity, Acceptance, Trust (low arousal, positive valence)
  * Top-left: Anger, Rage, Vigilance (high arousal, negative valence)
  * Bottom-left: Sadness, Grief, Pensiveness (low arousal, negative valence)
- Sarcastic patterns often show "emotional leaps" across opposite wheel sections
"""

file_path = 'The Void.mp3'  # or the full path to your file

# 1) Figure out where the last 120 s start:
total_dur = librosa.get_duration(filename=file_path)
start_sec = max(0, total_dur - 120.0)

# 2) Use ffmpeg to grab that segment as a 16 kHz mono WAV in-memory:
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
plt.title('Spectrogram (Last 120 s)')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

# --- Emotional Flow Scatter (4D) ---
frame_len = int(0.05 * sr)   # 50 ms frames
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
plt.xlabel('Arousal Proxy (norm RMS)')
plt.ylabel('Valence Proxy (norm Spectral Centroid)')
plt.title('Emotional Flow Scatter (Last 120 s)')
cbar = plt.colorbar()
cbar.set_label('Time (s)')
plt.tight_layout()
plt.show()

# Define emotion regions based on the emotion wheel
def add_emotion_wheel_overlay(ax, rms_range=(0,1), cent_range=(0,1), alpha=0.2):
    # Define colors for different emotion regions
    emotions = {
        'joy_ecstasy': {'color': 'yellow', 'center': (0.75, 0.75), 'radius': 0.25},
        'serenity_acceptance': {'color': 'lightgreen', 'center': (0.25, 0.75), 'radius': 0.25},
        'anger_rage': {'color': 'red', 'center': (0.75, 0.25), 'radius': 0.25},
        'sadness_grief': {'color': 'blue', 'center': (0.25, 0.25), 'radius': 0.25},
        'surprise_amazement': {'color': 'cyan', 'center': (0.5, 0.85), 'radius': 0.25},
        'fear_terror': {'color': 'darkgreen', 'center': (0.15, 0.5), 'radius': 0.25},
        'disgust_loathing': {'color': 'magenta', 'center': (0.5, 0.15), 'radius': 0.25},
        'anticipation_vigilance': {'color': 'orange', 'center': (0.85, 0.5), 'radius': 0.25},
    }
    
    # Add emotion regions
    for emotion, props in emotions.items():
        circle = plt.Circle(props['center'], props['radius'], color=props['color'], alpha=alpha, label=emotion)
        ax.add_patch(circle)
        ax.text(props['center'][0], props['center'][1], emotion.replace('_', '/'), 
                ha='center', va='center', fontsize=8, color='black')
    
    return ax

# Create a figure with two subplots - one normal and one with emotion wheel
plt.figure(figsize=(15, 6))

# Original scatter plot
ax1 = plt.subplot(1, 2, 1)
scatter = ax1.scatter(rms_n, cent_n, s=sizes, c=colors, cmap='viridis')
ax1.set_xlabel('Arousal Proxy (norm RMS)')
ax1.set_ylabel('Valence Proxy (norm Spectral Centroid)')
ax1.set_title('Emotional Flow Scatter (Last 120 s)')
cbar = plt.colorbar(scatter, ax=ax1)
cbar.set_label('Time (s)')

# Scatter plot with emotion wheel overlay
ax2 = plt.subplot(1, 2, 2)
scatter2 = ax2.scatter(rms_n, cent_n, s=sizes, c=colors, cmap='viridis', zorder=10)
ax2 = add_emotion_wheel_overlay(ax2)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel('Arousal Proxy (norm RMS)')
ax2.set_ylabel('Valence Proxy (norm Spectral Centroid)')
ax2.set_title('Emotional Mapping to Emotion Wheel')
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Time (s)')

plt.tight_layout()

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
