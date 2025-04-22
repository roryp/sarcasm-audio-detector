# Sarcasm Audio Detector

This project analyzes audio files to detect potential sarcasm in speech using acoustic features like pitch variations, timing, and emphasis patterns.

## Features

- **Spectrogram Visualization**: Shows frequency patterns over time, revealing the "acoustic fingerprint" of speech
- **Emotional Flow Visualization**: Tracks emotional dynamics that correlate with sarcastic speech using a 4D scatter plot
- **Emotion Wheel Mapping**: Maps acoustic features to emotional categories
- **Sarcasm Detection**: Identifies potential sarcasm by finding patterns with rapid emotional transitions

## Requirements

- Python 3.x
- Libraries: numpy, soundfile, librosa, matplotlib, subprocess
- ffmpeg (for audio processing)

## Usage

```python
# Run the script with your audio file
python sarcasm_detector.py
```

The script is currently configured to analyze "The Void.mp3" but can be modified to work with any audio file.