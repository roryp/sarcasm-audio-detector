# Sarcasm Audio Detector

A tool for analyzing acoustic patterns in speech to detect sarcasm.
Created: April 22, 2025

## The Story of Sarcasm in Sound

In a world where text analysis misses crucial emotional cues, we set out to capture what machines couldn't see: the subtle art of sarcasm. Our journey began with a simple question - what if we could visualize the sound of someone rolling their eyes?

### The Acoustic Fingerprint of Irony

Sarcasm isn't just what you say - it's how you say it. Our analysis reveals that sarcastic speech leaves distinctive trails in sound:

- Exaggerated pitch variations that contradict literal meaning
- Unique timing patterns that create emotional dissonance
- Emphasis markers that signal the speaker's true intent

Using "The Void.mp3" as our case study, we discovered that spectrograms can reveal these invisible patterns - showing the frequency distributions and energy patterns that betray when someone's words don't match their meaning.

### A Journey Through Emotional Space

Our breakthrough came with the development of the Emotional Flow Visualization - a 4D scatter plot that maps speech along dimensions that matter:

- **Intensity** (X-axis): Where vocal emphasis reveals emotional arousal
- **Tone** (Y-axis): Where brightness exposes valence mismatches 
- **Emphasis** (Size): Where larger points highlight the key moments of sarcastic expression
- **Time** (Color): Where the pattern of emotional shifts unfolds before your eyes

### What We Discovered

The patterns were unmistakable. When someone speaks sarcastically, they create distinctive signatures in emotional space:
- Clusters of high arousal paired with unexpected valence
- Temporal transitions that differ dramatically from sincere speech
- Exaggerated emphasis patterns that signal emotional incongruence

These acoustic patterns allow us to detect what once seemed impossible to capture - the subtle art of saying one thing while meaning another.

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

The script will generate visualizations that reveal the hidden emotional landscape of speech, showing you what the words alone could never tell.

## License

[MIT](LICENSE)