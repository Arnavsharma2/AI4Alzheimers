# CogniSense Dataset Guide

This document provides information on acquiring datasets for each modality in the CogniSense system.

---

## Speech Analysis Dataset

### Primary: DementiaBank (Pitt Corpus)

**Description**: Audio recordings of picture description task (Cookie Theft) from AD patients and controls.

**Access**:
1. Visit: https://dementia.talkbank.org/
2. Register for free academic access
3. Download the Pitt Corpus
4. Extract to `data/raw/speech/dementia_bank/`

**Statistics**:
- ~500 participants (AD + controls)
- Audio + transcripts available
- Cookie Theft picture description task

**Alternative: ADReSS Challenge Dataset**
- https://luzs.gitlab.io/adress/
- Pre-processed and balanced
- Requires registration

### Data Format Expected
```
data/raw/speech/
├── audio/
│   ├── participant_001.wav
│   ├── participant_002.wav
│   └── ...
└── transcripts/
    ├── participant_001.txt
    ├── participant_002.txt
    └── ...
```

---

## Eye Tracking Dataset

### Option 1: Synthetic Data (Recommended for Hackathon)

Use our built-in synthetic generator:
```python
from src.data_processing.synthetic_data_generator import EyeTrackingGenerator

gen = EyeTrackingGenerator()
data = gen.generate_sequence(is_alzheimers=False)
```

### Option 2: Real Eye Tracking Data

**Research Papers to Cite**:
- "Eye Tracking Detects Disconjugate Eye Movements Associated with Structural Traumatic Brain Injury"
- Search for published AD eye-tracking datasets on OSF, Zenodo

**Data Format Expected**:
```
data/raw/eye_tracking/
├── participant_001.csv  # Columns: timestamp, x, y
├── participant_002.csv
└── ...
```

---

## Typing Dynamics Dataset

### Synthetic Data (Recommended)

AD-specific typing datasets are extremely rare. We recommend using synthetic data:

```python
from src.data_processing.synthetic_data_generator import TypingDynamicsGenerator

gen = TypingDynamicsGenerator()
data = gen.generate_sequence(is_alzheimers=True)
```

### Data Format Expected
```
data/raw/typing/
├── participant_001.npy  # Shape: (sequence_length, 5 features)
├── participant_002.npy
└── ...
```

Features:
1. Flight time (ms)
2. Dwell time (ms)
3. Digraph latency (ms)
4. Error indicator (0/1)
5. Pause duration (ms)

---

## Clock Drawing Dataset

### Option 1: Kaggle Datasets

Search Kaggle for "clock drawing test" datasets:
- https://www.kaggle.com/datasets (search: "clock drawing")

### Option 2: Synthetic Generation

```python
from src.data_processing.synthetic_data_generator import ClockDrawingGenerator

gen = ClockDrawingGenerator(image_size=224)
img = gen.generate_image(is_alzheimers=False)
```

### Option 3: Request from Research Labs

Contact labs publishing AD research and request de-identified clock drawings.

**Data Format Expected**:
```
data/raw/clock_drawing/
├── AD/
│   ├── clock_001.jpg
│   ├── clock_002.jpg
│   └── ...
└── Control/
    ├── clock_101.jpg
    ├── clock_102.jpg
    └── ...
```

---

## Gait Analysis Dataset

### Primary: mHealth Dataset (UCI)

**Description**: Accelerometer and gyroscope data during various activities including walking.

**Access**:
1. Visit: https://archive.ics.uci.edu/ml/datasets/mhealth+dataset
2. Download dataset (9 MB)
3. Extract to `data/raw/gait/mhealth/`

**Statistics**:
- 10 volunteers
- 12 types of activities
- Accelerometer + gyroscope sensors
- 50 Hz sampling rate

### Alternative: Synthetic Generation

```python
from src.data_processing.synthetic_data_generator import GaitDataGenerator

gen = GaitDataGenerator(sampling_rate=50)
data = gen.generate_sequence(is_alzheimers=False, duration=10)
```

**Data Format Expected**:
```
data/raw/gait/
├── participant_001.npy  # Shape: (3, num_samples) - x, y, z accelerometer
├── participant_002.npy
└── ...
```

---

## Quick Start: Generate All Synthetic Data

For rapid prototyping and hackathon submission, generate a complete synthetic dataset:

```python
from src.data_processing.synthetic_data_generator import generate_synthetic_dataset

# Generate 200 samples (100 AD, 100 Control)
dataset = generate_synthetic_dataset(
    num_samples=200,
    ad_ratio=0.5,
    output_dir='data/processed/'
)
```

This generates data for all modalities with realistic AD vs Control differences.

---

## Data Preprocessing

Once you have raw data, preprocess it using:

```bash
# Preprocess all modalities
python src/data_processing/preprocess_speech.py
python src/data_processing/preprocess_eye.py
python src/data_processing/preprocess_typing.py
python src/data_processing/preprocess_drawing.py
python src/data_processing/preprocess_gait.py
```

Or run all at once:
```bash
python src/data_processing/preprocess_all.py
```

---

## Labels File

Create a labels file mapping participant IDs to diagnosis:

```csv
participant_id,diagnosis,age,sex
001,AD,72,F
002,Control,68,M
003,AD,79,F
...
```

Save as: `data/labels.csv`

---

## Notes for Hackathon Judges

For the AI 4 Alzheimer's Hackathon submission:

1. **Synthetic Data is Acceptable**: For modalities where real data is scarce (eye tracking, typing), we use synthetic data with published AD characteristics
2. **Unique Datasets**: We combine speech (DementiaBank), gait (mHealth), and clock drawing (Kaggle) with synthetic modalities
3. **Justification**: Our synthetic generators are based on peer-reviewed research on AD biomarkers
4. **Transparency**: All generation code is open and reproducible

---

## Citation

If using these datasets, please cite:

**DementiaBank**:
```
Becker, J. T., Boiler, F., Lopez, O. L., Saxton, J., & McGonigle, K. L. (1994).
The natural history of Alzheimer's disease: description of study cohort and accuracy of diagnosis.
Archives of Neurology, 51(6), 585-594.
```

**mHealth**:
```
Banos, O., Garcia, R., Holgado-Terriza, J. A., Damas, M., Pomares, H., Rojas, I., ... & Villalonga, C. (2014).
mHealthDroid: a novel framework for agile development of mobile health applications.
In Ambient assisted living and daily activities (pp. 91-98). Springer, Cham.
```

---

## Questions?

For dataset questions, see:
- Main README: `README.md`
- GitHub Issues: https://github.com/Arnavsharma2/AI4Alzheimers/issues
