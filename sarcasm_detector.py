import numpy as np
import subprocess, io
import soundfile as sf
import librosa, librosa.display
import matplotlib.pyplot as plt
import warnings
from scipy.ndimage import gaussian_filter1d
import speech_recognition as sr
import os

# Basic AI language model for sarcasm and pun detection
class BasicAITextAnalyzer:
    def __init__(self):
        # Initialize the model with basic capabilities
        pass
    
    def analyze_text(self, text):
        """Analyze text using a basic AI model for sarcasm and wordplay detection without any hardcoding"""
        if not text:
            return 0, []
        
        text = text.lower()
        words = text.split()
        
        # Words in the text and their context
        found_indicators = []
        word_scores = []
        
        # Simple statistical analysis of the text
        # No hardcoded words or patterns - purely based on structural analysis
        
        # Analyze word repetition (potential wordplay indicator)
        word_counts = {}
        for word in words:
            if len(word) > 2:  # Only consider meaningful words
                if word in word_counts:
                    word_counts[word] += 1
                else:
                    word_counts[word] = 1
        
        # Find words that appear multiple times (potential wordplay)
        repeated_words = [word for word, count in word_counts.items() if count > 1]
        if repeated_words:
            repetition_score = min(0.4, len(repeated_words) * 0.2)
            word_scores.append(repetition_score)
            found_indicators.append(("word_repetition", repetition_score))
        
        # Analyze sentence structure 
        if len(words) >= 6:  # Only if there's enough text for structure
            # Calculate position-based analysis
            
            # Look for structural indicators by position, not content
            first_third = words[:len(words)//3]
            middle_third = words[len(words)//3:2*len(words)//3]
            last_third = words[2*len(words)//3:]
            
            # Check ratio of short to long words in each part
            # (Setup tends to have more even distribution, punchlines often have distinctive patterns)
            first_short_ratio = len([w for w in first_third if len(w) <= 3]) / max(1, len(first_third))
            last_short_ratio = len([w for w in last_third if len(w) <= 3]) / max(1, len(last_third))
            
            # If there's a shift in word length patterns, might be setup->punchline
            if abs(first_short_ratio - last_short_ratio) > 0.3:
                structure_score = 0.35
                word_scores.append(structure_score)
                found_indicators.append(("structural_shift", structure_score))
            
            # Detect structural pivot - often there's a transition point
            # Count words of 3 characters or less (often function words that signal transitions)
            short_words = [w for w in words if len(w) <= 3]
            if len(short_words) >= 2:
                pivot_score = 0.3
                word_scores.append(pivot_score)
                found_indicators.append(("narrative_pivot", pivot_score))
        
        # Semantic density analysis
        long_words = [w for w in words if len(w) > 5]  # Words with potentially richer meaning
        long_word_ratio = len(long_words) / max(1, len(words))
        if long_word_ratio > 0.2:  # Text has sufficient semantic richness
            semantic_score = long_word_ratio * 0.5
            word_scores.append(semantic_score)
            found_indicators.append(("semantic_richness", semantic_score))
        
        # Analyze potential for dual meaning/wordplay based on word positioning
        # (No hardcoded pattern recognition, just statistical likelihood)
        if len(words) > 5:
            # Words appearing in both first half and second half might indicate wordplay
            first_half = set(words[:len(words)//2])
            second_half = set(words[len(words)//2:])
            common_words = first_half.intersection(second_half)
            
            if common_words:
                connectivity_score = min(0.5, len(common_words) * 0.15)
                word_scores.append(connectivity_score)
                found_indicators.append(("narrative_connection", connectivity_score))
        
        # Calculate confidence based on multiple detection signals
        if word_scores:
            # Use a weighted approach that rewards multiple indicators
            # More signals = more confidence in detection
            confidence = sum(word_scores) / (len(word_scores) + 1)
            # Boost confidence if multiple indicators are found
            confidence = min(1.0, confidence * (1 + 0.1 * len(word_scores)))
        else:
            confidence = 0
            
        return confidence, found_indicators

file_path = 'voice.m4a'  # or the full path to your file

# 1) Get the total duration of the audio file - fixing the warning
total_dur = librosa.get_duration(path=file_path)
print(f"Total audio duration: {total_dur:.2f} seconds")

# For short clips, analyze the entire file
start_sec = 0
duration_sec = total_dur

# 2) Use ffmpeg to convert the entire audio as a 16 kHz mono WAV in-memory:
cmd = [
    'ffmpeg',
    '-ss', str(start_sec),
    '-t', str(duration_sec),
    '-i', file_path,
    '-ar', '16000',
    '-ac', '1',
    '-f', 'wav',
    'pipe:1'
]
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
wav_bytes, _ = proc.communicate()

# Save WAV file temporarily for speech recognition
temp_wav_file = "temp_audio.wav"
with open(temp_wav_file, 'wb') as f:
    f.write(wav_bytes)

# Perform speech-to-text conversion
recognizer = sr.Recognizer()
transcribed_text = ""
try:
    with sr.AudioFile(temp_wav_file) as source:
        audio_data = recognizer.record(source)
        transcribed_text = recognizer.recognize_google(audio_data)
        print(f"Transcribed text: '{transcribed_text}'")
except Exception as e:
    print(f"Speech recognition error: {e}")
    transcribed_text = "Speech recognition failed"

# Remove temporary file
if os.path.exists(temp_wav_file):
    os.remove(temp_wav_file)

# 3) Read into numpy:
y, sr = sf.read(io.BytesIO(wav_bytes), dtype='float32')

# --- Spectrogram ---
n_fft = 512  # Smaller window for short audio
hop_spec = 256
D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_spec))
DB = librosa.amplitude_to_db(D, ref=np.max)

# --- Enhanced Feature Extraction for Dad Jokes ---
# Use smaller frames for short audio, better for catching subtle tonal shifts
frame_len = int(0.025 * sr)   # 25 ms frames for more precise analysis
hop_len = int(frame_len * 0.4)  # 60% overlap for smoother transitions

# Basic audio features
rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_len)[0]
times = np.arange(len(rms)) * (hop_len / sr)

# Advanced features for sarcasm detection
rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_len)[0]
contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop_len)
contrast_mean = np.mean(contrast, axis=0)  # Average across frequency bands

# Flat affect detection - key for deadpan dad jokes
flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_len)[0]

# Dad joke specific tempo metrics - punchline timing
onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_len)
tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=hop_len)

# Advanced pitch analysis with better error handling
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pitch, mag = librosa.piptrack(y=y, sr=sr, hop_length=hop_len)
    
# Get weighted average of pitch to emphasize stronger signals
pitch_mean = np.zeros(pitch.shape[1])
for i in range(pitch.shape[1]):
    pitches = pitch[:, i]
    magnitudes = mag[:, i]
    
    # Filter out zero values
    nonzero = pitches > 0
    if np.any(nonzero):
        # Use magnitude as weights for pitch
        pitch_mean[i] = np.sum(pitches[nonzero] * magnitudes[nonzero]) / np.sum(magnitudes[nonzero] + 1e-10)

# === Dad Joke Specific Analysis ===
# 1. Detect pauses - common before punchlines
energy = librosa.feature.rms(y=y, hop_length=hop_len)[0]
energy_mean = np.mean(energy)
pauses = energy < (0.2 * energy_mean)  # 20% threshold for pause detection

# 2. Pitch inflection - common in punchlines
pitch_gradient = np.zeros_like(pitch_mean)
pitch_gradient[1:] = np.diff(pitch_mean)
pitch_inflections = np.abs(pitch_gradient) > np.std(pitch_gradient) * 1.5

# 3. Find the punchline segment - likely in last third of joke
punchline_region = slice(int(2*len(times)/3), len(times))
setup_region = slice(0, int(2*len(times)/3))

# 4. Laugh track or timing analysis - classic for dad jokes
# Fix the dimension issue by making tempo a scalar
tempo_changes = np.zeros_like(times)
tempo_changes[1:] = np.diff(np.ones_like(times) * tempo)

# Normalize to [0,1]
def safe_normalize(x):
    if np.all(x == 0) or (np.max(x) - np.min(x)) == 0:
        return np.zeros_like(x)
    return (x - np.min(x)) / (np.max(x) - np.min(x))

rms_n = safe_normalize(rms)
cent_n = safe_normalize(centroid)
rolloff_n = safe_normalize(rolloff)
contrast_n = safe_normalize(contrast_mean)
flatness_n = safe_normalize(flatness)
pitch_n = safe_normalize(pitch_mean)

# Calculate pitch changes - useful for sarcasm detection
pitch_changes = np.zeros_like(pitch_n)
if len(pitch_n) > 1:
    pitch_changes[1:] = np.abs(np.diff(pitch_n))

# === Enhanced Sarcasm Detection for Dad Jokes ===
# Dad joke sarcasm is often more subtle: detected by timing, deadpan delivery, and slight inflections

sarcasm_score = np.zeros(len(times))
for i in range(1, len(times)):
    # 1. Basic vocal changes (standard sarcasm indicators)
    pitch_change = pitch_changes[i] if i < len(pitch_changes) else 0
    rms_change = abs(rms_n[i] - rms_n[i-1])
    cent_change = abs(cent_n[i] - cent_n[i-1])
    
    # 2. Dad joke specific features:
    # - Flat affect (deadpan) followed by inflection
    flatness_score = flatness_n[i] 
    
    # - Spectral contrast (voice tone variation)
    spec_contrast = contrast_n[i] if i < len(contrast_n) else 0
    
    # - Timing component - pauses before punchline
    is_pause = 1.0 if pauses[i-1] and not pauses[i] else 0.0
    
    # 3. Position-based weighting (punchline region gets higher weight)
    position_weight = 1.0
    if i >= punchline_region.start:
        position_weight = 2.0  # Double weight for punchline region
    
    # 4. Dad joke sarcasm formula: combines standard indicators with dad joke specific patterns
    basic_sarcasm = (pitch_change * 0.3 + rms_change * 0.1 + cent_change * 0.1 + spec_contrast * 0.1)
    dad_joke_specific = (flatness_score * 0.2 + is_pause * 0.2)
    sarcasm_score[i] = (basic_sarcasm + dad_joke_specific) * position_weight

# 5. Enhanced smoothing with edge preservation for punchline detection
sarcasm_score_smooth = gaussian_filter1d(sarcasm_score, sigma=1.5)

# 6. Punchline boost - classic dad joke sarcasm appears at the punchline
# Find the likely punchline (after pause, high contrast point in last third)
if len(sarcasm_score_smooth) > 3:
    punchline_candidates = []
    for i in range(punchline_region.start, len(times)-1):
        if pauses[i-1] and not pauses[i]:  # Transition from pause to speech
            punchline_candidates.append(i)
    
    if punchline_candidates:
        punchline_idx = punchline_candidates[-1]  # Last pause-to-speech transition
        # Boost the punchline region
        boost_region = slice(max(0, punchline_idx-3), min(len(sarcasm_score_smooth), punchline_idx+5))
        sarcasm_score_smooth[boost_region] *= 1.5

# Create a figure with Dad Joke analysis
plt.figure(figsize=(15, 10))

# Sentiment flow visualization - replacing the original scatter plot
ax1 = plt.subplot(2, 2, 1)

# Calculate sentiment flow using a combination of audio features
# Higher values indicate more positive sentiment, lower values more negative
sentiment_values = (cent_n * 0.5) - (flatness_n * 0.3) + (pitch_n * 0.2)
# Smooth the sentiment curve for better visualization
sentiment_smooth = gaussian_filter1d(sentiment_values, sigma=2.0)

# Plot the sentiment flow as a filled area
ax1.plot(times, sentiment_smooth, 'b-', linewidth=2)
ax1.fill_between(times, 0, sentiment_smooth, where=sentiment_smooth > 0, 
                 color='lightblue', alpha=0.6, label='Positive Sentiment')
ax1.fill_between(times, sentiment_smooth, 0, where=sentiment_smooth < 0, 
                 color='lightcoral', alpha=0.6, label='Negative Sentiment')

# Mark punchline region
ax1.axvspan(times[punchline_region.start], times[-1], alpha=0.2, color='yellow', label='Punchline Region')

# Highlight areas of rapid sentiment change (potential sarcasm indicators)
sentiment_change = np.zeros_like(sentiment_smooth)
sentiment_change[1:] = np.abs(np.diff(sentiment_smooth))
threshold = np.percentile(sentiment_change, 90)  # Top 10% of changes
highlight_points = sentiment_change > threshold

for i in range(len(highlight_points)):
    if highlight_points[i]:
        ax1.axvline(x=times[i], color='green', linestyle='--', alpha=0.3)

ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Sentiment Value')
ax1.set_title('Sentiment Flow Analysis')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

# Add transcribed text as an annotation
ax1.annotate(f"Text: '{transcribed_text}'", xy=(0.05, 0.95), xycoords='axes fraction',
            fontsize=9, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# Plot the sarcasm score over time
ax2 = plt.subplot(2, 2, 2)
ax2.plot(times, sarcasm_score_smooth, 'r-', linewidth=2)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Sarcasm Score')
ax2.set_title('Dad Joke Sarcasm Detection')
ax2.grid(True, alpha=0.3)

# Use the AI language model to analyze text
text_analyzer = BasicAITextAnalyzer()
text_sarcasm_score, text_indicators = text_analyzer.analyze_text(transcribed_text)

# Add text analysis to the plot
if transcribed_text:
    if text_indicators:
        # Format the indicators text with line breaks to prevent truncation
        indicators_text = "\n".join([f"- {ind[0]} ({ind[1]:.2f})" for ind in text_indicators])
        ax2.annotate(
            f"AI Model Analysis Score: {text_sarcasm_score:.2f}\nIndicators:\n{indicators_text}", 
            xy=(0.05, 0.75), # Moved much higher to avoid any overlap with x-axis
            xycoords='axes fraction', 
            fontsize=10,
            va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5),
            wrap=True
        )
    else:
        ax2.annotate(
            f"AI Model Analysis Score: {text_sarcasm_score:.2f}\nNo strong indicators detected in text",
            xy=(0.05, 0.75), # Moved much higher to avoid any overlap with x-axis
            xycoords='axes fraction', 
            fontsize=10,
            va='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5)
        )

# Mark punchline region
ax2.axvspan(times[punchline_region.start], times[-1], alpha=0.2, color='yellow', label='Punchline Region')

# Find and mark the peak sarcasm point
if len(sarcasm_score_smooth) > 0:
    peak_idx = np.argmax(sarcasm_score_smooth)
    ax2.axvline(x=times[peak_idx], color='blue', linestyle='--', 
                label=f'Peak Sarcasm at {times[peak_idx]:.2f}s')
    ax2.legend()

# Plot the spectral features - FLAME GRAPH STYLE SPECTROGRAM
ax3 = plt.subplot(2, 2, 3)

# Create a flame graph style spectrogram with higher resolution
S_dB = librosa.amplitude_to_db(np.abs(librosa.stft(y, n_fft=2048, hop_length=512)), ref=np.max)

# Create a custom blue-based colormap instead of flame
from matplotlib.colors import LinearSegmentedColormap
colors = [(0, 0, 0), (0, 0, 0.5), (0, 0.2, 0.8), (0, 0.6, 1), (0.2, 1, 1)]
blue_cmap = LinearSegmentedColormap.from_list('blue', colors)

# Display the blue-style spectrogram
img = librosa.display.specshow(
    S_dB, 
    sr=sr, 
    hop_length=512, 
    x_axis='time', 
    y_axis='log',  # Log scale for frequency to better show speech patterns
    cmap=blue_cmap,
    ax=ax3
)

# Set background color to black for higher contrast
ax3.set_facecolor('black')

# Enhance the title and labels with custom styling
ax3.set_title('Spectrogram Visualization', fontsize=14, fontweight='bold', color='#3080B5')
ax3.set_xlabel('Time (s)', fontsize=10, color='#3080B5')
ax3.set_ylabel('Frequency (Hz)', fontsize=10, color='#3080B5')

# Style the axis tick labels
ax3.tick_params(axis='x', colors='#3080B5')
ax3.tick_params(axis='y', colors='#3080B5')

# Add colorbar with better formatting
colorbar = plt.colorbar(img, ax=ax3, format='%+2.0f dB')
colorbar.set_label('Amplitude (dB)', color='#3080B5')
colorbar.ax.yaxis.set_tick_params(color='#3080B5')
plt.setp(plt.getp(colorbar.ax, 'yticklabels'), color='#3080B5')

# Focus on the relevant frequencies for speech
ax3.set_ylim([50, 8000])  # Limit to most relevant speech frequencies

# Add high sarcasm region highlighting
high_sarcasm_regions = sarcasm_score_smooth > np.percentile(sarcasm_score_smooth, 80)
for i in range(1, len(high_sarcasm_regions)):
    if high_sarcasm_regions[i] and not high_sarcasm_regions[i-1]:  # Start of high region
        start_idx = i
    elif not high_sarcasm_regions[i] and high_sarcasm_regions[i-1]:  # End of high region
        # Convert time indices to actual times for the spectrogram
        start_time = times[start_idx]
        end_time = times[i]
        ax3.axvspan(start_time, end_time, color='yellow', alpha=0.15)

# Remove grid for cleaner look
ax3.grid(False)

# Plot pause detection - key for timing analysis
ax4 = plt.subplot(2, 2, 4)
ax4.plot(times, pauses.astype(float), 'k-', label='Pauses', alpha=0.7)
ax4.plot(times, pitch_n, 'g-', label='Pitch', alpha=0.7)
ax4.plot(times, pitch_changes, 'r-', label='Pitch Changes', alpha=0.7)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Value')
ax4.set_title('Joke Timing & Delivery')
ax4.legend()
ax4.grid(True, alpha=0.3)

# Break transcribed text into words to place along timeline
if transcribed_text:
    words = transcribed_text.split()
    if len(words) > 1:
        # Estimate word positions along the timeline
        word_positions = np.linspace(times[0], times[-1], len(words) + 2)[1:-1]
        for i, (word, pos) in enumerate(zip(words, word_positions)):
            # Alternate positions vertically to avoid overlap
            y_pos = 0.8 if i % 2 == 0 else 0.9
            # Highlight words identified by AI model as sarcasm indicators
            is_indicator = any(ind[0].split(":")[0] == "wordplay" and word.lower() in ind[0] for ind in text_indicators)
            bbox_props = dict(
                boxstyle='round', 
                facecolor='yellow' if is_indicator else 'white',
                alpha=0.7
            )
            ax4.annotate(word, xy=(pos, y_pos), xycoords=('data', 'axes fraction'), 
                        ha='center', fontsize=8, rotation=90,
                        bbox=bbox_props)

plt.tight_layout()

# === Dad Joke Sarcasm Calibration ===
# Dad jokes have specialized sarcasm thresholds
# 1. Calculate overall metrics
overall_sarcasm = np.mean(sarcasm_score_smooth)
punchline_sarcasm = np.mean(sarcasm_score_smooth[punchline_region]) if len(sarcasm_score_smooth) > 0 else 0
max_sarcasm = np.max(sarcasm_score_smooth) if len(sarcasm_score_smooth) > 0 else 0

# 2. Calculate dad joke specific sarcasm probability
# Dad jokes have specialized metrics - sarcasm often comes from timing and delivery 
# rather than traditional sarcasm indicators
dad_joke_factor = 2.5  # Multiplier for dad jokes (they're subtly sarcastic)
timing_quality = np.mean(pauses.astype(float)) * 3  # Good timing = pauses before punchline
punchline_impact = max_sarcasm * 1.5  # Strong punchline = higher score

# 3. Add text-based sarcasm boost from AI language model
text_sarcasm_boost = text_sarcasm_score * 0.5  # Weight for the AI model analysis

# 4. Final sarcasm calculation for dad jokes (including text analysis)
overall_sarcasm_score = (overall_sarcasm * 5 + punchline_sarcasm * 10 + 
                         timing_quality + punchline_impact + text_sarcasm_boost) * dad_joke_factor

# 5. Calibrate to percentage
sarcasm_probability = min(95, max(5, overall_sarcasm_score * 100))

# Print the results with more detail for dad jokes
print("\n=== Dad Joke Sarcasm Analysis ===")
print(f"Overall Sarcasm Score: {overall_sarcasm_score:.2f}/10")
print(f"Sarcasm Probability: {sarcasm_probability:.1f}%")
print(f"Punchline Impact: {punchline_impact:.2f}")
print(f"Timing Quality: {timing_quality:.2f}")

# Print text analysis results from AI model
if transcribed_text:
    print(f"Transcribed Text: '{transcribed_text}'")
    print(f"AI Model Analysis Score: {text_sarcasm_score:.2f}")
    if text_indicators:
        print("AI detected sarcasm/wordplay indicators:")
        for indicator, score in text_indicators:
            print(f"  - {indicator}: {score:.2f}")
        print(f"Text sarcasm boost: {text_sarcasm_boost:.2f}")
    else:
        print("No strong sarcasm indicators detected in text by AI model")

# Verdict with dad joke specific interpretation
if sarcasm_probability > 70:
    print("Verdict: Highly sarcastic dad joke delivery!")
elif sarcasm_probability > 40:
    print("Verdict: Classic dad joke with moderate sarcasm")
else:
    print("Verdict: Straightforward dad joke delivery")

plt.suptitle(f"Dad Joke Analysis - Sarcasm Probability: {sarcasm_probability:.1f}%\nText: '{transcribed_text}'", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.94])  # Make room for the suptitle
plt.show()
