# DroneRF Loader Guide

This document explains how the DroneRF loader works, how to use it for training, and how to reuse/extend it later.

## Where the code lives

- Loader module: `data/data.py`
- Main load function: `load_dronerf_dataframe(...)`

## What the loader does

`load_dronerf_dataframe` is the main entrypoint. It does the following:

1. Finds all DroneRF `.rar` archives under the dataset root.
2. Parses class code and band from archive names (supports both `RF` and `FR` prefixes).
3. Extracts archives (Python backend first, CLI fallback next).
4. Reads numeric signal files from extracted folders.
5. Computes compact statistical + FFT features per archive.
6. Pairs `H` and `L` band archives into one sample row.
7. Adds labels for three tasks:
   - binary target (`background` vs `drone`)
   - family target (`background`, `bebop`, `ar`, `phantom`)
   - mode target (10-class code-level label)

The returned object is a pandas DataFrame (easy for training, analysis, and export).

## Main function

```python
from data.data import load_dronerf_dataframe

df = load_dronerf_dataframe(
    data_root=None,                # defaults to data/DroneRF
    extract_to=None,               # defaults to data/DroneRF/_extracted
    extract_archives=True,
    force_reextract=False,
    max_values_per_archive=200_000,
    fft_bins=2048,
    include_raw_signal=False,
)
```

### Parameters

- `data_root`: path to DroneRF root folder.
- `extract_to`: where extracted archive content is stored.
- `extract_archives`: extract `.rar` files when `True`.
- `force_reextract`: re-extract even if extracted folders already exist.
- `max_values_per_archive`: cap signal values per archive (memory/speed control).
- `fft_bins`: FFT bins used for spectral features.
- `include_raw_signal`: attach raw signal arrays in output DataFrame.

## Output DataFrame structure

Each row is one paired sample (`H` + `L`).

### Metadata columns

- `sample_id`
- `code`
- `family_name`
- `mode_name`
- `target_binary`
- `target_family`
- `target_mode`

### Source/provenance columns

- `h_archive_name`, `l_archive_name`
- `h_archive_index`, `l_archive_index`
- `h_archive_path`, `l_archive_path`
- `h_source_dir`, `l_source_dir`

### Feature columns

For both `h_` and `l_` prefixes:

- `signal_length`
- `mean`, `std`, `min`, `max`, `median`
- `q25`, `q75`, `iqr`
- `rms`, `abs_mean`, `energy`
- `fft_peak_bin`, `fft_peak_power`, `fft_mean_power`, `fft_entropy`

Extra cross-band features:

- `delta_rms`
- `ratio_rms`

## Label mapping used by loader

| Code  | Family     | target_family | target_binary | target_mode |
|-------|------------|---------------|---------------|-------------|
| 00000 | background | 0             | 0             | 0           |
| 10000 | bebop      | 1             | 1             | 1           |
| 10001 | bebop      | 1             | 1             | 2           |
| 10010 | bebop      | 1             | 1             | 3           |
| 10011 | bebop      | 1             | 1             | 4           |
| 10100 | ar         | 2             | 1             | 5           |
| 10101 | ar         | 2             | 1             | 6           |
| 10110 | ar         | 2             | 1             | 7           |
| 10111 | ar         | 2             | 1             | 8           |
| 11000 | phantom    | 3             | 1             | 9           |

## Training helper functions

### NumPy + train/test split

```python
from data.data import load_dronerf_dataframe, dataframe_to_numpy

df = load_dronerf_dataframe()
arrays = dataframe_to_numpy(
    dataframe=df,
    target_column="target_family",  # or target_binary / target_mode
    test_size=0.2,
    random_state=42,
    stratify=True,
)

X_train, y_train = arrays["X_train"], arrays["y_train"]
X_test, y_test = arrays["X_test"], arrays["y_test"]
```

### PyTorch Dataset/DataLoader

```python
from data.data import load_dronerf_dataframe, dataframe_to_torch_dataloader

df = load_dronerf_dataframe()
loader = dataframe_to_torch_dataloader(
    dataframe=df,
    target_column="target_family",
    batch_size=32,
    shuffle=True,
)
```

## Extraction behavior and fallback order

When extraction is enabled, the loader tries:

1. Python `rarfile`
2. CLI tools: `unar`, then `unrar`, then `bsdtar`

If all fail, it raises a clear error message with install hints.

## Reuse tips

- If you add new DroneRF class codes, update mapping dictionaries in `data/data.py`.
- If archive naming pattern changes, update `_ARCHIVE_NAME_RE`.
- If you want other features, extend `_signal_features(...)`.
- For large experiments, reduce `max_values_per_archive` for faster iteration.
- For full-fidelity runs, increase `max_values_per_archive` or set it to `None`.

## Common issues

- `No DroneRF .rar files found`:
  - Check `data_root` points to the dataset folder.
- `Failed to extract archive`:
  - Install extraction backend (for example `pip install rarfile` and `brew install unar`).
- `No supported numeric files found`:
  - Confirm extracted content contains supported data files (`csv`, `txt`, `dat`, `npy`, `npz`, `mat`).

## Minimal workflow for team members

```python
from data.data import load_dronerf_dataframe, dataframe_to_numpy

# 1) Load data to DataFrame
# 2) Convert to model arrays
# 3) Train classifier

df = load_dronerf_dataframe()
arrays = dataframe_to_numpy(df, target_column="target_family")
```

This is the recommended default for quick model experiments.
