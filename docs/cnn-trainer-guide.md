# CNN Multitask Trainer Guide

This guide explains how to train the DroneRF classifier with the multitask CNN trainer.

## Files

- Trainer script: `cnn.py`
- Data loader: `data/data.py`
- Loader guide: `docs/dronerf-loader-guide.md`

## What the Trainer Does

The trainer in `cnn.py`:

1. Loads DroneRF with raw signals enabled via `load_dronerf_dataframe(..., include_raw_signal=True)`.
2. Builds fixed-length 2-channel inputs from `h_raw_signal` and `l_raw_signal`.
3. Splits data into train/test sets (stratified by mode when feasible).
4. Trains one shared 1D CNN encoder with two heads:
   - family head (4 classes)
   - mode head (10 classes)
5. Evaluates both heads with accuracy and macro F1.
6. Saves model checkpoint and per-head artifacts.

## Requirements

Install dependencies from project root:

```bash
uv sync
```

Run commands through uv so project dependencies are always available:

```bash
uv run python cnn.py --help
```

## Quick Start

Run a quick multitask training pass with post-training accuracy test:

```bash
uv run python cnn.py --epochs 3 --batch-size 16 --max-values-per-archive 50000 --device auto --auto-test
```

This single run trains and reports metrics for both targets:

- `family` (4-class)
- `mode` (10-class)

## Mac M4 Device Support (MPS)

The trainer supports Apple Silicon acceleration.

- `--device auto`:
  - Uses `cuda` if available.
  - Else uses `mps` on Apple Silicon.
  - Else falls back to `cpu`.
- `--device mps`: force Apple GPU.

Examples:

```bash
uv run python cnn.py --device auto
```

```bash
uv run python cnn.py --device mps
```

## Main Arguments

### Data and Input

- `--data-root`: path to DroneRF root (default: `data/DroneRF`)
- `--test-size`: test split ratio (default: `0.20`)
- `--extract-archives` / `--no-extract-archives`
- `--force-reextract` / `--no-force-reextract`
- `--max-values-per-archive`: cap values per archive (default: `200000`)
- `--fft-bins`: FFT bins used by loader (default: `2048`)
- `--window-features` / `--no-window-features`: enable archive windowing (default: enabled)
- `--window-size`: window length used by loader (default: `4096`)
- `--window-stride`: stride for windowing (default: half-window when windowing enabled)
- `--sequence-length`: final CNN signal length per channel after pad/truncate (default: equals `--window-size`)
- `--loader-cache-dir`: custom loader cache directory
- `--loader-max-workers`: loader worker count (`0` means auto)
- `--show-loading-progress` / `--no-show-loading-progress`
- `--use-loader-cache` / `--no-use-loader-cache`
- `--refresh-loader-cache` / `--no-refresh-loader-cache`

### Model

- `--conv-channels`: comma-separated Conv1d channel sizes (default: `32,64,128`)
- `--kernel-size`: convolution kernel size (default: `7`)
- `--dropout`: dropout rate (default: `0.30`)

### Training

- `--epochs`: number of epochs (default: `50`)
- `--batch-size`: mini-batch size (default: `32`)
- `--learning-rate`: Adam LR (default: `0.001`)
- `--weight-decay`: Adam weight decay (default: `0.0001`)
- `--early-stopping-patience`: stop after no combined F1 improvement (default: `10`)
- `--family-loss-weight`: weight for family head loss (default: `1.0`)
- `--mode-loss-weight`: weight for mode head loss (default: `1.0`)
- `--use-class-weights` / `--no-use-class-weights`
- `--auto-test` / `--no-auto-test`: run final evaluation and save test artifacts

### Runtime and Reproducibility

- `--device`: `auto`, `cpu`, `cuda`, or `mps`
- `--seed`: global random seed (default: `42`)
- `--deterministic` / `--no-deterministic`
- `--normalize-signals` / `--no-normalize-signals`

### Output Paths

- `--model-dir` (default: `models`)
- `--output-dir` (default: `results`)

## Output Artifacts

The trainer writes one shared model and per-head outputs.

Model:

- `models/cnn_multitask.pt`

Validation outputs:

- `results/cnn_metrics_family.json`
- `results/cnn_confusion_family.csv`
- `results/cnn_predictions_family.csv`
- `results/cnn_metrics_mode.json`
- `results/cnn_confusion_mode.csv`
- `results/cnn_predictions_mode.csv`

When `--auto-test` is enabled, additional files are written:

- `results/cnn_test_metrics_family.json`
- `results/cnn_test_confusion_family.csv`
- `results/cnn_test_predictions_family.csv`
- `results/cnn_test_metrics_mode.json`
- `results/cnn_test_confusion_mode.csv`
- `results/cnn_test_predictions_mode.csv`

## Recommended Commands

Quick smoke test:

```bash
uv run python cnn.py --epochs 3 --batch-size 16 --max-values-per-archive 50000 --device auto --auto-test
```

Better baseline run:

```bash
uv run python cnn.py --epochs 40 --batch-size 32 --device auto --auto-test
```

Increase sequence detail:

```bash
uv run python cnn.py --window-size 8192 --sequence-length 8192 --epochs 40 --batch-size 16 --device auto
```

Emphasize mode head during optimization:

```bash
uv run python cnn.py --mode-loss-weight 1.5 --family-loss-weight 1.0 --epochs 50 --device auto
```

Warm-cache repeated experiments:

```bash
uv run python cnn.py --use-loader-cache --loader-max-workers 0
```

Force a fresh loader rebuild:

```bash
uv run python cnn.py --refresh-loader-cache --loader-max-workers 1
```

## Notes on Metrics

- Family and mode are reported separately (accuracy, macro F1, confusion matrix, predictions).
- Early stopping uses combined macro F1 (average of family macro F1 and mode macro F1).
- Mode is typically harder because it is a 10-class task with class imbalance.

## Troubleshooting

### MPS requested but unavailable

- Verify Apple Silicon and recent PyTorch build.
- Use `--device auto` for safe fallback.

### Training is slow or memory usage is high

- Reduce `--sequence-length`.
- Reduce `--window-size`.
- Reduce `--batch-size`.
- Reduce `--max-values-per-archive`.
- Use smaller channels (for example `--conv-channels 16,32,64`).

### Accuracy stays low on quick runs

- Increase epochs (quick runs are mainly pipeline checks).
- Keep `--auto-test` on and compare family vs mode metrics.
- Tune `--mode-loss-weight` for better mode focus.
- Try larger `--sequence-length` if memory allows.
