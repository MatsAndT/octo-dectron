import os
import re
import csv
import numpy as np
from glob import glob

# ----------------------------
# CONFIG
# ----------------------------
CACHE_PATH = "cache/dronerf_dataset.npz"
FFT_SIZE = 4096


# ----------------------------
# FILE PARSING
# ----------------------------
def parse_filename(name):
    m = re.match(r"(\d+)([LH])_(\d+)\.csv", name)
    if not m:
        return None
    full_code, band, seg = m.groups()
    mode = full_code[-2:]
    return mode, band, seg


# ----------------------------
# LOAD SIGNAL
# ----------------------------
def load_signal(path):
    with open(path, newline="", encoding="utf-8") as f:
        row = next(csv.reader(f))
    return np.array(row, dtype=np.float32)


# ----------------------------
# FFT
# ----------------------------
def fft(x):
    x = x * np.hanning(len(x))
    return np.abs(np.fft.rfft(x, n=FFT_SIZE))


# ----------------------------
# BUILD FILE INDEX
# ----------------------------
def build_index(data_dir):
    files = glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

    grouped = {}
    for f in files:
        parsed = parse_filename(os.path.basename(f))
        if not parsed:
            continue

        mode, band, seg = parsed
        grouped.setdefault((mode, seg), {})[band] = f

    samples = []
    modes = sorted({k[0] for k in grouped})
    mode_map = {m: i for i, m in enumerate(modes)}

    for (mode, seg), bands in grouped.items():
        if "L" in bands and "H" in bands:
            samples.append((mode_map[mode], bands["L"], bands["H"]))

    return samples, mode_map


# ----------------------------
# FEATURE EXTRACTOR
# ----------------------------
def spectral_peaks(x, k=5):
    idx = np.argpartition(x, -k)[-k:]
    idx = idx[np.argsort(x[idx])[::-1]]
    return x[idx], idx  # return the k largest peaks magnitudes and their frequencies

def extract_features(L, H, mode="mlp"):
    fL = fft(L)
    fH = fft(H)

    if mode == "cnn":
        return np.concatenate([fL, fH])

    if mode == "mlp":
        peaks_L, freqs_L = spectral_peaks(fL)
        peaks_H, freqs_H = spectral_peaks(fH)

        return np.concatenate([
            peaks_L, freqs_L,
            peaks_H, freqs_H,

            [fL.min(), fL.mean(), fL.std(), fL.max()],
            [fH.min(), fH.mean(), fH.std(), fH.max()],
        ]).astype(np.float32)

    raise ValueError("mode must be 'cnn' or 'mlp'")


# ----------------------------
# BUILD DATASET (NO PYTORCH)
# ----------------------------
def build_dataset(data_dir, mode="mlp"):
    samples, mode_map = build_index(data_dir)

    X = []
    y = []

    for label, l_path, h_path in samples:
        L = load_signal(l_path)
        H = load_signal(h_path)

        n = min(len(L), len(H))
        L, H = L[:n], H[:n]

        X.append(extract_features(L, H, mode))
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), mode_map


# ----------------------------
# CACHE HANDLING
# ----------------------------
def load_or_build(data_dir, mode="mlp", use_cache=True):
    import os
    import numpy as np

    if use_cache and os.path.exists(CACHE_PATH):
        print("Loading cached dataset...")

        data = np.load(CACHE_PATH, allow_pickle=True)

        cached_mode = str(data["mode"].item() if hasattr(data["mode"], "item") else data["mode"])

        if cached_mode == mode:
            X = data["X"]
            y = data["y"].astype(np.int64)

            mode_map = data["mode_map"].item() if isinstance(data["mode_map"], np.ndarray) else data["mode_map"]

            feature_names = (
                list(data["feature_names"])
                if "feature_names" in data.files
                else get_feature_names(mode)
            )

            print("Cache loaded:")
            print("X:", X.shape, "y:", y.shape)
            print("Classes:", np.unique(y))

            return X, y, {v: k for k, v in mode_map.items()}, feature_names

    print("Building dataset from CSV...")

    X, y, mode_map = build_dataset(data_dir, mode)
    feature_names = get_feature_names(mode)

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

    np.savez(
        CACHE_PATH,
        X=X,
        y=y,
        mode_map=mode_map,
        feature_names=np.array(feature_names, dtype=object),
        mode=mode
    )

    return X, y, {v: k for k, v in mode_map.items()}, feature_names


# ----------------------------
# FEATURE NAMES (optional helper)
# ----------------------------
def get_feature_names(mode="mlp"):
    if mode == "cnn":
        return None

    return [
        "fL_peak1", "fL_peak2", "fL_peak3", "fL_peak4", "fL_peak5",
        "fL_freq1", "fL_freq2", "fL_freq3", "fL_freq4", "fL_freq5",
        "fH_peak1", "fH_peak2", "fH_peak3", "fH_peak4", "fH_peak5",
        "fH_freq1", "fH_freq2", "fH_freq3", "fH_freq4", "fH_freq5",

        "fL_min", "fL_mean", "fL_std", "fL_max",
        "fH_min", "fH_mean", "fH_std", "fH_max",
    ]


# ----------------------------
# MAIN TEST
# ----------------------------
if __name__ == "__main__":
    data_dir = "DroneRF"

    X, y, label_map, feature_names = load_or_build(data_dir, mode="mlp")

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Label map (numeric -> mode string):", label_map)