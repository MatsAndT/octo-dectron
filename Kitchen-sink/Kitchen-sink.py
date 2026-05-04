import os
import numpy as np
from glob import glob
from sklearn.preprocessing import StandardScaler

# ----------------------------
# 1. Load signal
# ----------------------------
def load_signal(file_path):
    return np.loadtxt(file_path)

# ----------------------------
# 2. Convert to frequency domain
# ----------------------------
def compute_fft(signal):
    fft = np.fft.rfft(signal)              # real FFT
    magnitude = np.abs(fft)                # magnitude spectrum
    log_mag = np.log1p(magnitude)          # log scaling
    return log_mag

# ----------------------------
# 3. Process one segment (L + H)
# ----------------------------
def process_segment(L_path, H_path):
    L_time = load_signal(L_path)
    H_time = load_signal(H_path)

    L_freq = compute_fft(L_time)
    H_freq = compute_fft(H_time)

    # Concatenate into one feature vector
    features = np.concatenate([L_freq, H_freq])
    return features

# ----------------------------
# 4. Build dataset
# ----------------------------
def build_dataset(data_dir):
    X = []
    y = []

    modes = os.listdir(data_dir)

    for label, mode in enumerate(modes):
        mode_path = os.path.join(data_dir, mode)

        if not os.path.isdir(mode_path):
            continue

        # Find all L files
        L_files = sorted(glob(os.path.join(mode_path, "*_L.*")))

        for L_path in L_files:
            H_path = L_path.replace("_L", "_H")

            if not os.path.exists(H_path):
                continue  # skip if missing pair

            features = process_segment(L_path, H_path)

            X.append(features)
            y.append(label)

    return np.array(X), np.array(y)

# ----------------------------
# 5. Normalize dataset
# ----------------------------
def normalize_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

# ----------------------------
# 6. Main
# ----------------------------
if __name__ == "__main__":
    data_dir = "data"  # change this

    X, y = build_dataset(data_dir)
    X, scaler = normalize_data(X)

    print("Dataset shape:", X.shape)
    print("Labels shape:", y.shape)

    # Now ready for training
    # Example:
    # model.fit(X, y)