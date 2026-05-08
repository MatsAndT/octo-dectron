import os
import re
import csv
import numpy as np
from glob import glob

# --- KONFIGURASJON ---
CACHE_DIR = "cache"
FFT_SIZE = 4096

def get_cache_path(mode):
    """Genererer unikt filnavn for hver modus så de ikke overskriver hverandre."""
    return os.path.join(CACHE_DIR, f"dronerf_dataset_{mode}.npz")

# --- FIL-PARSING OG LASTING ---

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

# --- SIGNALBEHANDLING ---

def fft(x):
    x = x * np.hanning(len(x))
    return np.abs(np.fft.rfft(x, n=FFT_SIZE))

def spectral_peaks(x, k=5):
    idx = np.argpartition(x, -k)[-k:]
    idx = idx[np.argsort(x[idx])[::-1]]
    return x[idx], idx

def extract_features(L, H, mode="mlp"):
    fL = fft(L)
    fH = fft(H)

    if mode == "cnn":
        # Returnerer hele spekteret (ca 4098 verdier)
        return np.concatenate([fL, fH])

    if mode == "mlp":
        # Returnerer kun utvalgte features (28 verdier)
        peaks_L, freqs_L = spectral_peaks(fL)
        peaks_H, freqs_H = spectral_peaks(fH)

        return np.concatenate([
            peaks_L, freqs_L,
            peaks_H, freqs_H,
            [fL.min(), fL.mean(), fL.std(), fL.max()],
            [fH.min(), fH.mean(), fH.std(), fH.max()],
        ]).astype(np.float32)

    raise ValueError("mode must be 'cnn' or 'mlp'")

# --- DATASETT-BYGGING ---

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

def build_dataset(data_dir, mode="mlp"):
    samples, mode_map = build_index(data_dir)
    X, y = [], []

    for label, l_path, h_path in samples:
        L, H = load_signal(l_path), load_signal(h_path)
        n = min(len(L), len(H))
        X.append(extract_features(L[:n], H[:n], mode))
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), mode_map

# --- HOVEDFUNKSJON FOR LASTING ---

def load_or_build(data_dir, mode="mlp", use_cache=True):
    cache_path = get_cache_path(mode)
    
    if use_cache and os.path.exists(cache_path):
        print(f"Loading cached dataset from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        
        # Sjekker for sikkerhets skyld at cachen faktisk inneholder riktig modus
        if str(data["mode"]) == mode:
            X = data["X"]
            y = data["y"].astype(np.int64)
            mode_map = data["mode_map"].item()
            feature_names = data["feature_names"] if "feature_names" in data.files else None
            
            print(f"Cache loaded successfully. X shape: {X.shape}")
            return X, y, {v: k for k, v in mode_map.items()}, feature_names

    print(f"Building new {mode.upper()} dataset from CSV...")
    X, y, mode_map = build_dataset(data_dir, mode)
    feature_names = get_feature_names(mode)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(
        cache_path,
        X=X, y=y, mode_map=mode_map,
        feature_names=np.array(feature_names, dtype=object),
        mode=mode
    )

    return X, y, {v: k for k, v in mode_map.items()}, feature_names

def get_feature_names(mode="mlp"):
    if mode == "cnn": return None
    return [
        "fL_peak1", "fL_peak2", "fL_peak3", "fL_peak4", "fL_peak5",
        "fL_freq1", "fL_freq2", "fL_freq3", "fL_freq4", "fL_freq5",
        "fH_peak1", "fH_peak2", "fH_peak3", "fH_peak4", "fH_peak5",
        "fH_freq1", "fH_freq2", "fH_freq3", "fH_freq4", "fH_freq5",
        "fL_min", "fL_mean", "fL_std", "fL_max",
        "fH_min", "fH_mean", "fH_std", "fH_max",
    ]

if __name__ == "__main__":
    data_dir = ".DroneRF"
    # Nå kan du kjøre begge uten konflikt
    X_mlp, y_mlp, _, _ = load_or_build(data_dir, mode="mlp")
    X_cnn, y_cnn, _, _ = load_or_build(data_dir, mode="cnn")