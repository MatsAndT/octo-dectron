# octo-dectron

Classifies drone operating mode from RF signals using the [DroneRF dataset](https://ieee-dataport.org/open-access/dronerf). Two models are compared: an MLP operating on aggregated frequency-band statistics, and a 1D-CNN operating directly on windowed frequency sequences.

## Results

| Model | Test accuracy | Macro-F1 |
|---|---|---|
| Dummy Classifier (baseline) | 28.26 % | 0.09 |
| MLP (GridSearchCV) | 82.61 % | 0.82 |
| CNN (~11k params) | 71.74 % | 0.66 |

## Project structure

```
.
├── load_data.py        # Signal loading, windowing, FFT feature extraction, caching
├── mlp.py              # MLP training: pooling, dummy baseline, GridSearchCV
├── cnn.py              # CNN training: model definition, normalisation, training loop
├── cache/              # Cached .npz datasets (auto-generated, do not commit)
└── Final Assignment Rapport/   # Typst report source
```

## How it works

Raw DroneRF CSV files contain 10 million real-valued samples per file. The pipeline:

1. **Windowing** — each recording is split into overlapping windows of 65 536 samples (50 % hop), yielding ~300 windows per file.
2. **FFT** — each window is transformed via rFFT with a Hanning taper. The spectrum is divided into 32 frequency bands per channel (L and H), giving 64 band energies per window.
3. **MLP path** — the 300 windows are pooled to a flat 192-feature vector (mean + std + max per band). Features are scaled with `RobustScaler` fit on training data only.
4. **CNN path** — the (300, 64) matrix is fed directly to the model. Per-sample max-normalisation is applied.

## Setup

Requires Python ≥ 3.13 and [uv](https://github.com/astral-sh/uv).

```bash
uv sync
```

## Data

Download the DroneRF dataset and place it in a `.DroneRF/` folder at the project root. The expected filename pattern is `<BUI><band>_<segment>.csv`, e.g. `10000L_0.csv`.

The first run builds a cached `.npz` file in `cache/`. Subsequent runs load from cache.

## Running

**MLP** (trains dummy baseline + GridSearchCV MLP, prints results, shows confusion matrix):

```bash
uv run python mlp.py
```

**CNN** (trains model, saves `learning_curves.png`, shows confusion matrix):

```bash
uv run python cnn.py
```

## Dependencies

| Package | Purpose |
|---|---|
| `numpy` | Array operations, FFT |
| `scikit-learn` | MLP, GridSearchCV, metrics, preprocessing |
| `tensorflow` | 1D-CNN model |
| `matplotlib` | Plots and confusion matrices |
| `pandas` / `seaborn` | Exploratory data analysis |
