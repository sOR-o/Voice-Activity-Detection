# Voice Activity Detection (VAD) with Pitch and Energy Thresholding

This repository contains a Python script (`vad.py`) for performing Voice Activity Detection (VAD) on audio files using both pitch and energy-based thresholds. It uses the `librosa` and `opensmile` libraries for audio processing and extracts silence regions based on energy and pitch analysis.

## Requirements

Before running the script, ensure that the following dependencies are installed:

- Python 3.x
- `librosa`
- `opensmile`
- `argparse`
- `numpy`
- `matplotlib`
- `tqdm`
- `multiprocessing`

You can install these dependencies using pip:

```bash
pip install librosa opensmile numpy matplotlib tqdm
