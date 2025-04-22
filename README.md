# Sarcasm Audio Detector

A Python-based tool for analyzing audio patterns that can detect sarcasm in speech through acoustic analysis.

## Overview

This project uses audio signal processing to identify patterns in speech that correlate with sarcastic expression. By analyzing features like pitch, timing, and emphasis patterns that text analysis would miss, this tool visualizes the emotional flow that can reveal sarcasm.

## Features

- **Audio Spectrogram Analysis**: Visualizes frequency patterns over time that show distinctive sarcastic speech fingerprints
- **Emotional Flow Mapping**: Tracks speech dynamics along arousal and valence dimensions
- **Emotion Wheel Integration**: Maps acoustic patterns to standardized emotional categories
- **Sarcasm Pattern Detection**: Identifies characteristic emotional incongruence typical of sarcastic speech

## Requirements

- Python 3.6+
- Libraries:
  - numpy
  - soundfile
  - librosa
  - matplotlib
  - ffmpeg (external dependency)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sarcasm-audio-detector.git

# Navigate to project directory
cd sarcasm-audio-detector

# Install dependencies
pip install numpy soundfile librosa matplotlib
```

## Usage

Place your audio file in the project directory and run:

```python
python sarcasm_detector.py
```

The script will generate three visualizations:
1. A spectrogram of the audio's last 120 seconds
2. An emotional flow scatter plot mapping arousal vs. valence
3. An emotion wheel mapping showing how the speech patterns relate to emotional categories
4. An emotional transition analysis highlighting potential sarcasm indicators

## License

[MIT](LICENSE)