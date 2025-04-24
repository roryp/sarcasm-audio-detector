# Sarcasm Audio Detector

A sophisticated audio analysis tool for detecting sarcasm in speech with specialized support for dad jokes.
Created: April 22, 2025
Updated: April 24, 2025

## The Science of Sarcasm Detection

In a world where text analysis misses crucial emotional cues, we set out to capture what machines couldn't see: the subtle art of sarcasm. Our journey began with a simple question - what if we could visualize the sound of someone rolling their eyes?

### The Acoustic Fingerprint of Irony

Sarcasm isn't just what you say - it's how you say it. Our analysis reveals that sarcastic speech leaves distinctive trails in sound:

- Exaggerated pitch variations that contradict literal meaning
- Unique timing patterns that create emotional dissonance
- Emphasis markers that signal the speaker's true intent
- Deadpan delivery followed by subtle tonal shifts

Using spectrograms and advanced audio features, we can reveal these invisible patterns - showing the frequency distributions and energy patterns that betray when someone's words don't match their meaning.

### Dad Joke Sarcasm: A Special Case

Dad jokes present a unique challenge for sarcasm detection. Often delivered with intentional deadpan and subtle tonal shifts, their sarcasm is more nuanced than standard sarcastic speech. Our enhanced algorithm now specifically targets:

- The deadpan-to-punchline shift characteristic of dad joke delivery
- Subtle timing patterns with strategic pauses before punchlines
- Minimal but significant pitch inflections at key moments
- The contrast between setup and delivery in the punchline region

### A Journey Through Emotional Space

Our breakthrough came with the development of the Emotional Flow Visualization - a 4D scatter plot that maps speech along dimensions that matter:

- **Intensity** (X-axis): Where vocal emphasis reveals emotional arousal
- **Tone** (Y-axis): Where brightness exposes valence mismatches 
- **Emphasis** (Size): Where larger points highlight the key moments of sarcastic expression
- **Time** (Color): Where the pattern of emotional shifts unfolds before your eyes

### Advanced Features

The latest version implements several sophisticated analysis techniques:

- **Punchline Detection**: Automatic identification of joke punchlines through timing and tonal patterns
- **Spectral Contrast Analysis**: Detecting subtle variations in vocal tone characteristic of sarcastic speech
- **Deadpan Factor**: Measuring flat affect followed by tonal shifts - a hallmark of dad joke delivery
- **Temporal Weighting**: Enhanced sensitivity to sarcasm indicators in punchline regions

### Mapping the Emotional Landscape

Our most fascinating discovery was how sarcasm creates a unique pattern on the emotion wheel:
- Rapid oscillation between contrasting emotional states
- Movement between joy and disgust regions that signal insincerity
- Emotional "leaps" across opposite sections of the wheel

Each region of our emotional space corresponds to familiar feelings:
- Top-right: Joy, Ecstasy, Admiration (high arousal, positive valence)
- Bottom-right: Serenity, Acceptance, Trust (low arousal, positive valence)
- Top-left: Anger, Rage, Vigilance (high arousal, negative valence)
- Bottom-left: Sadness, Grief, Pensiveness (low arousal, negative valence)

Yet sarcastic speech refuses to follow the normal rules, jumping across these boundaries in ways that reveal the speaker's true intent.

## Technical Details

The system employs a multi-layered approach to sarcasm detection:

1. **Audio Preprocessing**: 
   - High-quality audio conversion using ffmpeg
   - Adaptive analysis window selection based on audio duration
   - Specialized frame sizing for short utterances like dad jokes

2. **Feature Extraction**:
   - Root Mean Square (RMS) energy for loudness detection
   - Spectral centroid for pitch and tonal qualities
   - Spectral flatness for deadpan delivery identification
   - Spectral contrast for emotional variance
   - Onset detection for timing analysis
   - Pitch tracking with magnitude weighting

3. **Dad Joke Analysis**:
   - Pause detection for timing analysis
   - Pitch inflection tracking
   - Punchline region identification
   - Vocal pattern changes at critical points

4. **Visualization Pipeline**:
   - Spectrogram generation for frequency analysis
   - 4D scatter plots for emotional flow visualization
   - Time-series analysis of sarcasm scores
   - Vocal characteristic plots with key indicators

## Requirements

- Python 3.6+
- Libraries:
  - numpy
  - soundfile
  - librosa
  - matplotlib
  - scipy
  - ffmpeg (external dependency)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/sarcasm-audio-detector.git

# Navigate to project directory
cd sarcasm-audio-detector

# Install dependencies
pip install numpy soundfile librosa matplotlib scipy
```

## Usage

### Basic Usage

Place your audio file in the project directory and update the file_path variable in sarcasm_detector.py:

```python
file_path = 'your_audio_file.m4a'  # Replace with your file
```

Then run:

```bash
python sarcasm_detector.py
```

### Example Files

The repository includes:
- `voice.m4a`: Sample dad joke for testing ("My wife told me to do lunges to stay in shape…I guess that's a big step forward.")
- `graph.py`: Helper module for additional visualizations
- `image.png`: Example output visualization

### Interpreting Results

The script outputs:
- **Overall Sarcasm Score**: Scale of 0-10 indicating sarcasm intensity
- **Sarcasm Probability**: Percentage likelihood of sarcastic intent (0-95%)
- **Punchline Impact**: Measurement of punchline effectiveness 
- **Timing Quality**: Analysis of pause placement and speech rhythm
- **Verdict**: Qualitative assessment of sarcasm level

### Visualizations

Four visualization panels are generated:
1. **Dad Joke Delivery Analysis**: Scatter plot showing intensity vs. tone with time-based coloring
2. **Sarcasm Detection Score**: Time-series plot of sarcasm intensity with punchline marker
3. **Dad Joke Vocal Characteristics**: Multiple feature plot showing volume, pitch, deadpan factor, and vocal expression
4. **Joke Timing & Delivery**: Analysis of pauses, pitch tracking and pitch changes

## Advanced Configuration

For advanced users, several parameters can be tuned:
- Frame length and hop size for different resolution analysis
- Pause detection threshold for timing sensitivity
- Punchline region definition (default is last third of audio)
- Sarcasm formula weighting for different speech styles

## Future Development

Planned enhancements:
- Multi-speaker sarcasm differentiation
- Cultural variation modeling for international sarcasm
- Real-time sarcasm detection API
- Expanded dad joke corpus for improved pattern recognition

## License

[MIT](LICENSE)

## Citation

If you use this tool in research, please cite:
```
@software{sarcasm_detector,
  author = {Your Name},
  title = {Sarcasm Audio Detector},
  year = {2025},
  url = {https://github.com/yourusername/sarcasm-audio-detector}
}
```