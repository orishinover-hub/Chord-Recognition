# Automatic Chord Recognition with Transformer Encoder

This repository contains an implementation of an **automatic chord recognition model** based on a **Transformer-Encoder architecture with a convolutional front-end**. The model is trained and evaluated on the [**GuitarSet dataset**](https://github.com/marl/guitarset), achieving **competitive performance** with state-of-the-art systems in the Music Information Retrieval (MIR) community.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)  
- [Motivation](#motivation)  
- [Dataset](#dataset)  
- [Vocabulary Modes](#vocabulary-modes)  
- [Methodology](#methodology)  
- [Installation](#installation)  
- [Hyperparameters](#hyperparameters)  
- [Data Augmentation](#data-augmentation)  
- [Evaluation Metrics & Analysis](#evaluation-metrics--analysis)  
- [Results](#results)  
- [Future Work](#future-work)  
- [References](#references)  

---

## 🎶 Project Overview
The goal of this project is to **automatically detect and classify chords from raw audio recordings** using deep learning.  

Applications include:
- Music education (automatic harmonic analysis)  
- Music production and composition  
- Music recommendation and retrieval  
- Musicology and large-scale harmonic analysis  

---

## 🎼 Dataset

### GuitarSet Overview
- **Source**: [GuitarSet](https://github.com/marl/guitarset), a widely used dataset for chord recognition and transcription.  
- **Content**:  
  - 360 excerpts (30 seconds each) of real guitar performances.  
  - Six guitars recorded with **hexaphonic pickups** (each string isolated).  
  - Paired with **time-aligned chord annotations** in `.jams` format.  

### Chord Annotations
- Stored in `.jams` files (JSON-based format).  
- Contain:
  - Start/end times for each chord.  
  - Chord labels (e.g., `C:maj`, `A:min7`, `N` for no chord).  
- Different vocabularies possible:  
  - **Full** (root + quality + bass).  
  - **No bass** (root + quality only).  
  - **Maj/min** (collapsed to 24 triads + `N`) for MIREX-style evaluation.  

### Loader Workflow
The function `load_guitarset_cqt` handles **audio + label preprocessing**.  

Steps:
1. **Scan all `.jams` files**  
   - Build the chord vocabulary using a `LabelEncoder` (maps chord name ↔ integer ID).  

2. **For each song**  
   - Load annotations (`lab_df`).  
   - Load the audio waveform (`y`).  
   - Compute the **CQT** spectrogram.  
   - Align chord labels to CQT frames.  
   - Append to dataset lists.  

3. **If augmentation is enabled**  
   - Apply pitch shifting (± semitones).  
   - Recompute CQT and labels.  
   - Add augmented version to dataset.  

4. **Return**  
   - `X_list`: list of CQT arrays.  
   - `Y_list`: list of integer chord sequences.  
   - `label_enc`: encoder for chord string ↔ index.  
   - `meta`: metadata (song name, augmentation type, shift).  

---

### Loader Code

```python
def load_guitarset_cqt(audio_dir=AUDIO_DIR, ann_dir=ANN_DIR, augment_cfg=augment_settings,
                       chord_layer="performed", vocab_mode="full"):
    """
    vocab_mode: "full"    (root+quality+bass)
                "no_bass" (root+quality only)
                "majmin"  (collapse to 24-class maj/min + N)
    """
    import random
    jams_files = sorted([f for f in os.listdir(ann_dir) if f.endswith(".jams")])
    all_chords = set()

    def normalize_labels(df):
        # Normalize to consistent labels
        df["chord"] = df["chord"].apply(normalize_major)
        if vocab_mode == "no_bass":
            df["chord"] = df["chord"].apply(drop_bass)
        elif vocab_mode == "majmin":
            df["chord"] = df["chord"].apply(to_majmin)
        return df

    loaded, skipped = 0, 0

    # Build vocabulary
    for jf in jams_files:
        lab_df = jams_to_lab_df(os.path.join(ann_dir, jf), chord_layer=chord_layer)
        lab_df = normalize_labels(lab_df)
        all_chords.update(lab_df["chord"].unique())
    label_enc = LabelEncoder().fit(sorted(all_chords))

    X_list, Y_list, meta = [], [], []
    for jf in jams_files:
        stem = os.path.splitext(jf)[0]
        wav_path = find_audio_path(audio_dir, stem)
        jams_path = os.path.join(ann_dir, jf)
        if not os.path.exists(wav_path):
            print(f"[skip] audio missing for {stem}")
            skipped += 1
            continue

        lab_df = jams_to_lab_df(jams_path, chord_layer=chord_layer)
        lab_df = normalize_labels(lab_df)

        y, _ = librosa.load(wav_path, sr=SR)

        # Compute original CQT
        C_db = compute_cqt(y)
        if augment_cfg.get("enable", False):
            if np.random.rand() < 0.5: C_db = time_mask(C_db, max_width=16)
            if np.random.rand() < 0.5: C_db = freq_mask(C_db, max_width=8)

        labels = align_labels_to_frames(lab_df, C_db.shape[1])
        Y_int = label_enc.transform(labels)
        X_list.append(C_db); Y_list.append(Y_int)
        meta.append(dict(song=stem, aug="orig", shift=0))
        loaded += 1

        # Augmented (training only)
        if augment_cfg.get("enable", False):
            y_ps, lab_ps, semis = maybe_pitch_shift(y, SR, lab_df, augment_cfg)
            if semis != 0:
                lab_ps = normalize_labels(lab_ps)
                C_db_ps = compute_cqt(y_ps)
                labels_ps = align_labels_to_frames(lab_ps, C_db_ps.shape[1])
                Y_int_ps = label_enc.transform(labels_ps)
                X_list.append(C_db_ps); Y_list.append(Y_int_ps)
                meta.append(dict(song=stem, aug="pitch_shift", shift=semis))

    print(f"[vocab] mode={vocab_mode} | classes={len(label_enc.classes_)}")
    print(f"[summary] Loaded {loaded} tracks, Skipped {skipped}, Total={loaded+skipped}")
    print("[vocab] sample:", random.sample(list(label_enc.classes_), min(15, len(label_enc.classes_))))

    return X_list, Y_list, label_enc, meta
```

---

## 🎚 Vocabulary Modes

The loader supports **different chord vocabularies** depending on task complexity.  
You can switch the mode by setting `vocab_mode` when calling `load_guitarset_cqt`.  

### Example: Maj/Min Baseline (25 Classes)
```python
# Collapsed major/minor (24 + N = 25-class baseline)
X, Y, label_enc, meta = load_guitarset_cqt(vocab_mode="majmin")

n_bins = X[0].shape[0]
n_classes = len(label_enc.classes_)
print(f"[vocab] maj/min classes = {n_classes}")
```

- Matches **MIREX evaluation standard**.  
- Simplest vocabulary: major, minor, + `N` (no chord).  
- Good starting baseline.  

### Example: Full Vocabulary
```python
X, Y, label_enc, meta = load_guitarset_cqt(vocab_mode="full")
```
- Richest representation.  
- Includes root, chord quality, and bass note.  

### Example: No-Bass Vocabulary
```python
X, Y, label_enc, meta = load_guitarset_cqt(vocab_mode="no_bass")
```
- Intermediate difficulty.  
- Root + quality only (ignores bass inversion).  

### Recommended Workflow
- Start with `majmin` (simplest, 25 classes).  
- Once stable, upgrade to `no_bass`.  
- Finally, move to `full` for maximum realism.  

---

## 🏗 Methodology
- **Convolutional front-end** extracts local spectral patterns  
- **Positional encoding** provides temporal context  
- **Transformer Encoder** models long-range dependencies  
- **Linear output layer** produces frame-level chord predictions  

Training uses **cross-entropy loss (with label smoothing)**, **AdamW optimizer**, **warm-up + cosine LR scheduler**, and **dropout** for regularization.

---

## ⚙️ Installation

```bash
!pip install -q --upgrade   numpy scipy librosa jams scikit-learn audioread torch torchaudio

# Match google-colab’s pins (optional)
!pip install -q pandas==2.2.2 requests==2.32.3
```

---

## 🔧 Hyperparameters

### Model Parameters
- `d_model = 256`  
- `n_heads = 4`  
- `num_layers = 4`  
- `d_ff = 512`  
- `dropout = 0.1`  
- Conv1D kernel size = 5  

### Training Parameters
- Optimizer: **AdamW**  
- Scheduler: **warm-up + cosine decay**  
- Loss: **CrossEntropyLoss(ignore_index=-100, label_smoothing=0.05)**  
- Normalization: `"per_song"` (default), `"per_batch"`, `"dataset"`, `"none"`  
- Early stopping enabled  
- Evaluation decoders: `argmax`, `median (kernel=9)`, `viterbi`  

### Audio Resolution Parameters
- **SR = 22050**  
- **HOP = 512**  
- **BINS_PER_OCT = 12**  
- **N_BINS = 84**  
- **FMIN = C1 (~32.7 Hz)**  

---

## 🎛 Data Augmentation

To improve generalization, the project supports **pitch-shifting augmentation** during training.  
This simulates recordings in different keys by transposing audio up or down in semitone steps.  

### Settings
```python
augment_settings = dict(
    enable=True,  # Enable augmentation during training
    pitch_shift_choices=[-3, -2, -1, 0, 1, 2, 3],  # shift range in semitones
    prob=0.5      # 50% chance per example
)
```

- **Enable**:  
  - `True` → Augmentation applied (for training).  
  - `False` → No augmentation (for validation/testing).  

- **pitch_shift_choices**: Allowed semitone shifts (±3) to stay musically realistic.  
- **prob**: Probability per sample (default 0.5).  

### How to Enable/Disable
Edit the config cell in the notebook:

```python
# Constants and Configs
SR = 22050
HOP = 512
BINS_PER_OCT = 12
N_BINS = 84
FMIN = None  # defaults to C1

augment_settings = dict(
    enable=False,  # Set to True for training, False for val/test
    pitch_shift_choices=[-3, -2, -1, 0, 1, 2, 3],
    prob=0.5
)
```

---

## 📊 Evaluation Metrics & Analysis

Evaluating chord recognition models is tricky because chords are sequential, structured, and musically contextual.  
This project uses several complementary metrics:

### Frame-Level Accuracy
- The percentage of correctly predicted chords at the frame level.  
- Straightforward, but can overemphasize long-duration chords.  

### Weighted Chord Symbol Recall (WCSR)
- Standard in the **MIREX evaluation**.  
- Weights predictions by **segment duration** → better reflects perceptual relevance.  
- Ensures results are directly comparable with other systems in the MIR community.  

### Confusion Matrix Analysis
- Shows which chords are commonly confused with each other.  
- Useful for spotting systematic errors (e.g., C major vs. A minor).  

(Code to generate confusion matrix included in repo)

---

### Class Distribution Analysis
- Compares the frequency of each chord class in **ground truth vs. model predictions**.  
- Detects **class imbalance** or systematic bias in predictions.  

(Code to compare class distribution included in repo)

---

## 📊 Results

On the **GuitarSet test set**:
- **Weighted Chord Symbol Recall (WCSR)**: **80%**  
- **Frame-level Accuracy**: ~81%  

Competitive with state-of-the-art systems (81–83% WCSR).

---

## 🔮 Future Work
- Richer chord vocabularies (sevenths, suspensions, extended harmonies)  
- Multi-modal input (e.g., MIDI + audio)  
- Real-time / low-latency inference  
- Cross-dataset generalization  

---

## 📚 References
1. Li (2021) – CNN-HMM chord recognition  
2. Osmalskyj et al. (2018) – Neural network chord recognition  
3. Park et al. – Bi-directional Transformer for chord recognition  
