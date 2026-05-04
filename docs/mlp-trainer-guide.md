# MLP Trainer Guide

This guide explains how to train the DroneRF classifier with the MLP trainer script.

## Files

- Trainer script: `mlp.py`
- Data loader: `data/data.py`
- Loader guide: `docs/dronerf-loader-guide.md`

## What the Trainer Does

The trainer in `mlp.py`:

1. Loads DroneRF using `load_dronerf_dataframe`.
2. Builds train/test split with stratification.
3. Optionally scales features with `StandardScaler`.
4. Trains a PyTorch MLP with configurable hidden layers.
5. Evaluates with accuracy and macro F1.
6. Saves model and result artifacts.

## Requirements

Install dependencies from project root:

```bash
uv sync
```

## Quick Start

Run 4-class drone family classification:

```bash
uv run mlp.py --target family --epochs 30 --batch-size 32
```

Run binary detection (background vs drone):

```bash
uv run mlp.py --target binary --epochs 30
```

Run 10-class mode classification:

```bash
uv run mlp.py --target mode --epochs 40
```

## Mac M4 Device Support (MPS)

The trainer supports Apple Silicon acceleration.

- `--device auto`:
  - Uses `cuda` if available.
  - Else uses `mps` on Apple Silicon.
  - Else falls back to `cpu`.
- `--device mps`: force Apple GPU.

Examples:

```bash
uv run mlp.py --device auto
```

```bash
uv run mlp.py --device mps
```

## Main Arguments

### Data and Task

- `--data-root`: path to DroneRF root (default: `data/DroneRF`)
- `--target`: `binary`, `family`, or `mode` (default: `family`)
- `--test-size`: test split ratio (default: `0.20`)
- `--extract-archives` / `--no-extract-archives`
- `--force-reextract` / `--no-force-reextract`
- `--max-values-per-archive`: cap values per archive (default: `200000`)
- `--fft-bins`: FFT bins for feature extraction (default: `2048`)
- `--window-features` / `--no-window-features`: compute features on sliding windows per archive (mode defaults to enabled)
- `--window-size`: window length for windowed feature mode (default: `4096`)
- `--window-stride`: stride for windowed feature mode (default: equals `window-size`; mode uses 50% overlap when omitted)
- `--loader-cache-dir`: custom loader cache directory (default: loader-managed)
- `--loader-max-workers`: loader worker count (`0` means auto)
- `--show-loading-progress` / `--no-show-loading-progress`
- `--use-loader-cache` / `--no-use-loader-cache`
- `--refresh-loader-cache` / `--no-refresh-loader-cache`

### Model

- `--hidden-sizes`: comma-separated hidden layers (default: `256,128`)
- `--dropout`: dropout rate (default: `0.30`)

### Training

- `--epochs`: number of epochs (default: `50`)
- `--batch-size`: mini-batch size (default: `32`)
- `--learning-rate`: Adam LR (default: `0.001`)
- `--weight-decay`: Adam weight decay (default: `0.0001`)
- `--early-stopping-patience`: stop after no improvement (default: `10`)
- `--use-class-weights` / `--no-use-class-weights`
- `--auto-test` / `--no-auto-test`: run an extra final test pass and save test artifacts (default: off)

### Runtime and Reproducibility

- `--device`: `auto`, `cpu`, `cuda`, or `mps`
- `--seed`: global random seed (default: `42`)
- `--deterministic` / `--no-deterministic`
- `--scale-features` / `--no-scale-features`

### Output Paths

- `--model-dir` (default: `models`)
- `--output-dir` (default: `results`)

## Output Artifacts

For `--target family`, the trainer writes:

- `models/mlp_family.pt`
- `results/metrics_family.json`
- `results/confusion_family.csv`
- `results/predictions_family.csv`

The same naming pattern is used for `binary` and `mode`.

When `--auto-test` is enabled, the trainer additionally writes:

- `results/test_metrics_family.json`
- `results/test_confusion_family.csv`
- `results/test_predictions_family.csv`

## Recommended Commands

Fast smoke test:

```bash
uv run mlp.py --target family --epochs 3 --batch-size 16 --max-values-per-archive 50000
```

Run with progress bar enabled explicitly:

```bash
uv run mlp.py --show-loading-progress
```

Better baseline run:

```bash
uv run mlp.py --target family --epochs 50 --batch-size 32 --device auto
```

Baseline run with explicit post-training auto-test:

```bash
uv run mlp.py --target family --epochs 50 --batch-size 32 --device auto --auto-test
```

Mode task with class weighting:

```bash
uv run mlp.py --target mode --epochs 60 --use-class-weights --device auto
```

Mode task with window features (recommended for real training):

```bash
uv run mlp.py --target mode --window-features --window-size 4096 --window-stride 2048 --epochs 80 --use-class-weights --device auto
```

Warm-cache repeated experiments:

```bash
uv run mlp.py --target family --use-loader-cache --loader-max-workers 0
```

Force a fresh loader rebuild:

```bash
uv run mlp.py --refresh-loader-cache --loader-max-workers 1
```

## Notes on Class Imbalance

The DroneRF classes are imbalanced (especially Phantom), so macro F1 is tracked in addition to accuracy. Best model selection is based on validation macro F1.

## Troubleshooting

### MPS requested but unavailable

- Make sure you run on Apple Silicon and a recent PyTorch build.
- Try `--device auto` to fallback safely.

### Extraction is slow on first run

- First run extracts `.rar` files.
- Later runs reuse extracted content unless `--force-reextract` is set.

### Out-of-memory or very slow training

- Reduce `--max-values-per-archive`.
- Reduce `--batch-size`.
- Use fewer/lighter hidden layers (for example `--hidden-sizes 128,64`).
