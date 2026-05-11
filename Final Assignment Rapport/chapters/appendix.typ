#let appendix(body) = {
  set heading(numbering: "A.1:", supplement: [Vedlegg])
  counter(heading).update(0)
  body
}


#show: appendix
#align(center + horizon)[[Denne siden er blank med hensikt]]
#pagebreak()

= Vedlegg A — Dataloader
```python
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
```

#pagebreak()

= Vedlegg B — MLP
```python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils import compute_sample_weight
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay,
)
from load_data import load_or_build, N_BANDS


def pool_windows(X: np.ndarray) -> np.ndarray:
    """(n_files, n_windows, n_features) → (n_files, n_features * 3)"""
    mean = X.mean(axis=1)
    std  = X.std(axis=1)
    peak = X.max(axis=1)
    return np.concatenate([mean, std, peak], axis=1)


def run_dummy(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    print(f"Dummy test accuracy : {dummy.score(X_test, y_test):.4f}")
    print(f"Dummy test macro-F1 : {f1_score(y_test, dummy.predict(X_test), average='macro', zero_division=0):.4f}")



def run_mlp(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    sample_weights = compute_sample_weight("balanced", y_train)
    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("mlp", MLPClassifier(
            activation="relu", solver="adam", max_iter=5000,
            early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=50, learning_rate="adaptive", random_state=42,
        )),
    ])
    param_grid = {
        "mlp__hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32)],
        "mlp__alpha": [0.05, 0.1, 0.5, 1.0],
        "mlp__learning_rate_init": [0.001, 0.0005],
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train, mlp__sample_weight=sample_weights)
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    print("Beste parametere:", grid.best_params_)
    print(f"CV macro-F1: {grid.best_score_:.4f}")
    print(f"Test accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test macro-F1: {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.tight_layout(); plt.show()
    return best, grid


if __name__ == "__main__":
    data_dir = ".DroneRF"
    X_3d, y, label_map, _ = load_or_build(data_dir, mode="cnn")
    X = pool_windows(X_3d)
    run_dummy(X, y)
    run_mlp(X, y)
```

#pagebreak()

= Vedlegg C — CNN
```python
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

from load_data import load_or_build, MAX_WINDOWS, N_BANDS


def build_model(input_shape, num_classes):
    l2 = regularizers.L2(0.005)
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.GaussianNoise(0.05),
        layers.Conv1D(16, 7, activation="relu", padding="same", kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling1D(4),
        layers.Dropout(0.4),
        layers.Conv1D(32, 5, activation="relu", padding="same", kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.MaxPooling1D(4),
        layers.Dropout(0.4),
        layers.GlobalAveragePooling1D(),
        layers.Dense(32, activation="relu", kernel_regularizer=l2),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    data_dir = ".DroneRF"
    X, y, label_map, _ = load_or_build(data_dir, mode="cnn")
    num_classes = len(np.unique(y))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    def normalise(arr):
        m = np.max(np.abs(arr), axis=(1, 2), keepdims=True)
        m = np.where(m == 0, 1.0, m)
        return arr / m

    X_train = normalise(X_train)
    X_test  = normalise(X_test)

    class_weights = compute_class_weight(
        class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(class_weights))

    model = build_model((X_train.shape[1], X_train.shape[2]), num_classes)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=20, restore_best_weights=True)

    history = model.fit(
        X_train, y_train, batch_size=16, epochs=200,
        validation_data=(X_test, y_test),
        class_weight=class_weights, callbacks=[early_stop],
    )

    y_pred = np.argmax(model.predict(X_test), axis=1)
    print(classification_report(y_test, y_pred, zero_division=0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["loss"], label="train")
    axes[0].plot(history.history["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history.history["accuracy"], label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    plt.tight_layout()
    plt.savefig("learning_curves.png"); plt.show()

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
```
