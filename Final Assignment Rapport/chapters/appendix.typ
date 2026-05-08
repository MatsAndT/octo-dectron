#let appendix(body) = {
  set heading(numbering: "A.1:", supplement: [Vedlegg])
  counter(heading).update(0)
  body
}


#show: appendix
#align(center + horizon)[[Denne siden er blank med hensikt]]
#pagebreak()

= Vedlegg A 
== Dataloader
```python
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
    X_mlp, y_mlp, _, _ = load_or_build(data_dir, mode="mlp")
    X_cnn, y_cnn, _, _ = load_or_build(data_dir, mode="cnn")
```


== EDA av data til MLP

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from load_data import load_or_build

# -------------------------------------------------
# DATASET INFO (pandas)
# -------------------------------------------------
def print_dataset_info(X, y, label_map=None, feature_names=None):
    y = np.asarray(y).reshape(-1)

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    df = pd.DataFrame(X, columns=feature_names)
    df["label"] = y

    # map labels if provided
    if label_map is not None:
        df["label"] = df["label"].map(label_map)

    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"\nSamples:  {len(df)}")
    print(f"Features: {X.shape[1]}")

    print("\nClass distribution:")
    print(df["label"].value_counts())

    print("\nClass distribution (%):")
    print((df["label"].value_counts(normalize=True) * 100).round(2))

    print("\nFeature statistics:")
    print(df[feature_names].describe().T)

    print("=" * 60 + "\n")

    print(df.describe().T.sort_values("std", ascending=False))

    print(df.columns)


if __name__ == "__main__":

    data_dir = ".DroneRF"

    X, y, mode_map, feature_names = load_or_build(data_dir, mode="mlp")

    print_dataset_info(
        X,
        y,
        label_map=mode_map,
        feature_names=feature_names
    )
```

== MLP
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from load_data import load_or_build

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)

def filter_features(X_train, X_test, feature_names):
    """
    Fjerner manuelle features (max) og bruker VarianceThreshold 
    for å fjerne de med aller minst standardavvik.
    """
    df_train = pd.DataFrame(X_train, columns=feature_names)
    
    # 1. Fjern redundante features (de er identiske med peak1)
    to_drop = ["fL_max", "fH_max"]
    df_train = df_train.drop(columns=[c for c in to_drop if c in df_train.columns])
    
    # 2. Fjern features med svært lav varians
    selector = VarianceThreshold(threshold=1e-9) 
    selector.fit(df_train)
    
    # Lagre navnene på det vi beholder
    kept_features = df_train.columns[selector.get_support()].tolist()
    removed_features = [c for c in df_train.columns if c not in kept_features]
    
    print("\n--- FEATURE FILTERING ---")
    print(f"Fjernet redundante: {to_drop}")
    print(f"Fjernet pga lav varians: {removed_features}")
    print(f"Antall features beholdt: {len(kept_features)} av {len(feature_names)}")
    
    # Apply to train
    X_train_filtered = df_train[kept_features].values
    
    # Apply to test
    df_test = pd.DataFrame(X_test, columns=feature_names)
    df_test = df_test.drop(columns=[c for c in to_drop if c in df_test.columns])
    X_test_filtered = df_test[kept_features].values
    
    return X_train_filtered, X_test_filtered, kept_features

def run_dummy_classifier(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    print("\n--- DUMMY CLASSIFIER (Most Frequent) ---")
    print(f"Train Accuracy: {dummy.score(X_train, y_train):.4f}")
    print(f"Test Accuracy:  {dummy.score(X_test, y_test):.4f}")
    
    return dummy

def run_mlp_kitchen_sink(X, y):
    # Vi bruker alle originale features uten filtrering
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standard oppsett: ingen regularisering, standard lag
    pipe = Pipeline([
        ("scaler", StandardScaler()), 
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(100, 100), # Stor kapasitet
            alpha=0,                      # Ingen brems (ingen L2-straff)
            max_iter=2000,
            random_state=42
        ))
    ])

    pipe.fit(X_train, y_train)

    print("\n--- KITCHEN SINK MLP (Ufiltrert & Uregulert) ---")
    print(f"Train Accuracy: {pipe.score(X_train, y_train):.4f}")
    print(f"Test Accuracy:  {pipe.score(X_test, y_test):.4f}")
    
    return pipe

def run_mlp_pipeline(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Filtrer data etter split for å unngå data lekkasje
    X_train_filtered, X_test_filtered, filtered_names = filter_features(X_train, X_test, feature_names)


    # Pipeline med StandardScaler og MLP
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            activation="relu",
            solver="adam",
            max_iter=5000,
            
            # --- Early Stopping Parametere ---
            early_stopping=True,      # Denne var med da det gikk bra
            validation_fraction=0.15,  # Bruker 20% av trening til å sjekke når den skal stoppe
            n_iter_no_change=75,      # Gir den 50 runder på å forbedre seg før den stopper
            
            # --- Læringskontroll ---
            learning_rate="adaptive", 
            random_state=42
        ))
    ])

    param_grid = {
    # Vi vet at (16, 16) fungerer, så vi tester litt større varianter av "små" lag
    "mlp__hidden_layer_sizes": [
        (16, 16), 
        (32, 16), 
        (32, 32),
        (24, 24)
    ],
    
    # Siden 0.1 i alpha fungerte bra for å hindre overfitting, 
    # tester vi verdier rett rundt der
    "mlp__alpha": [0.05, 0.1, 0.2, 0.5],
    
    # Vi holder oss til 0.001, men legger til 0.005 for å se om raskere læring hjelper
    "mlp__learning_rate_init": [0.001, 0.005]
}

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_train_filtered, y_train)

    print("\nBEST PARAMETERS:")
    print(grid.best_params_)
    
    best_model = grid.best_estimator_

    # Sjekk resultater
    y_train_pred = best_model.predict(X_train_filtered)
    y_test_pred = best_model.predict(X_test_filtered)

    print(f"\nTRAIN Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"TEST Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
    
    print("\nCLASSIFICATION REPORT (TEST):")
    print(classification_report(y_test, y_test_pred, zero_division=0))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_test_pred,
        cmap="Blues",
        normalize=None
    )
    disp.ax_.set_title("Test-set Confusion Matrix")
    plt.tight_layout()
    plt.show()


    return best_model, grid

if __name__ == "__main__":

    data_dir = ".DroneRF" 

    X, y, mode_map, feature_names = load_or_build(data_dir, mode="mlp")

    run_dummy_classifier(X, y)
    run_mlp_kitchen_sink(X, y)
    model, grid = run_mlp_pipeline(X, y, feature_names)
```