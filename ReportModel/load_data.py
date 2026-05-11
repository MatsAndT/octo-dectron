import os
import re
import csv
import numpy as np
from glob import glob

CACHE_DIR = "cache"
WINDOW_SIZE = 65536   # ~64k sampler per vindu
HOP_SIZE    = WINDOW_SIZE // 2   # 50% overlapping → ~300 vinduer per fil
N_BANDS     = 32      # frekvensbånd per kanal
MAX_WINDOWS = 300     # pad / avkort til dette antallet

def get_cache_path(mode):
    if mode == "cnn":
        return os.path.join(CACHE_DIR, f"dronerf_dataset_cnn_w{WINDOW_SIZE}_b{N_BANDS}.npz")
    return os.path.join(CACHE_DIR, f"dronerf_dataset_{mode}.npz")

def parse_filename(name):
    m = re.match(r"(\d+)([LH])_(\d+)\.csv", name)
    if not m:
        return None
    full_code, band, seg = m.groups()
    model_prefix = full_code[:-2]
    label_code = full_code[-2:]
    if model_prefix == "000":
        label_code = "000"
    return model_prefix, label_code, band, seg

def load_signal(path):
    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.reader(f))
    return np.array(row, dtype=np.float32)

def _band_energies(window: np.ndarray) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(window * np.hanning(len(window)), n=WINDOW_SIZE))
    n_bins = len(spectrum)
    band_size = n_bins // N_BANDS
    return np.array(
        [spectrum[i * band_size : (i + 1) * band_size].mean() for i in range(N_BANDS)],
        dtype=np.float32,
    )

def _extract_windows(L: np.ndarray, H: np.ndarray) -> np.ndarray:
    n = min(len(L), len(H))
    rows = []
    start = 0
    while start + WINDOW_SIZE <= n:
        fl = _band_energies(L[start : start + WINDOW_SIZE])
        fh = _band_energies(H[start : start + WINDOW_SIZE])
        rows.append(np.concatenate([fl, fh]))
        start += HOP_SIZE
    if not rows:
        l_pad = np.zeros(WINDOW_SIZE, dtype=np.float32)
        h_pad = np.zeros(WINDOW_SIZE, dtype=np.float32)
        l_pad[:n] = L[:n]; h_pad[:n] = H[:n]
        rows.append(np.concatenate([_band_energies(l_pad), _band_energies(h_pad)]))
    return np.array(rows, dtype=np.float32)

def extract_features(L, H, mode="cnn"):
    if mode == "cnn":
        return _extract_windows(L, H)
    raise ValueError("mode must be 'cnn'")

def build_index(data_dir):
    files = glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
    grouped = {}
    for f in files:
        parsed = parse_filename(os.path.basename(f))
        if not parsed: continue
        model_prefix, label_code, band, seg = parsed
        grouped.setdefault((model_prefix, label_code, seg), {})[band] = f
    samples = []
    label_codes = sorted({key[1] for key in grouped})
    label_map = {label_code: i for i, label_code in enumerate(label_codes)}
    for (model_prefix, label_code, seg), bands in sorted(grouped.items()):
        if "L" in bands and "H" in bands:
            samples.append((label_map[label_code], bands["L"], bands["H"]))
    return samples, label_map

def build_dataset(data_dir):
    samples, mode_map = build_index(data_dir)
    X_list, y = [], []
    for label, l_path, h_path in samples:
        L, H = load_signal(l_path), load_signal(h_path)
        n = min(len(L), len(H))
        X_list.append(extract_features(L[:n], H[:n], mode="cnn"))
        y.append(label)
    n_features = N_BANDS * 2
    X_out = np.zeros((len(X_list), MAX_WINDOWS, n_features), dtype=np.float32)
    for i, windows in enumerate(X_list):
        n_w = min(len(windows), MAX_WINDOWS)
        X_out[i, :n_w] = windows[:n_w]
    return X_out, np.array(y, dtype=np.int64), mode_map

def load_or_build(data_dir, mode="cnn", use_cache=True):
    cache_path = get_cache_path(mode)
    if use_cache and os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        if str(data["mode"]) == mode:
            X = data["X"]
            y = data["y"].astype(np.int64)
            mode_map = data["mode_map"].item()
            return X, y, {v: k for k, v in mode_map.items()}, None
    X, y, mode_map = build_dataset(data_dir)
    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path, X=X, y=y, mode_map=mode_map, mode=mode)
    return X, y, {v: k for k, v in mode_map.items()}, None