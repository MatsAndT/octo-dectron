"""
MLP v2 — lærdommer fra CNN-forbedringen
========================================

Hva var galt med MLP v1:
  - 28 håndlagde features fra én enkelt 4096-punkts FFT per fil
  - Mye av informasjonen i signalet ble kastet bort
  - Scoring på accuracy er misvisende ved klasseubalanse
  - Ingen klassevekting (MLPClassifier har ikke class_weight)

Hva vi gjør nå (samme tankegang som CNN-fiksen):
  1. Bruker de 300 vinduene fra CNN-lasteren (64k samples, 50 % overlapping)
  2. Pooler på tvers av vinduene: mean + std + max per frekvensbånd → 192 features
     - mean:  typisk frekvensprofil for denne dronemodussignaturen
     - std:   temporal variasjon — ulike modi varierer ulikt over tid
     - max:   topp-energi i hvert bånd uavhengig av tidspunkt
  3. Scorer på macro-F1 i stedet for accuracy (riktig mål ved ubalanserte klasser)
  4. Klassevekting via sample_weight (sklearn MLPClassifier støtter ikke class_weight)
  5. RobustScaler i stedet for StandardScaler (tåler energitopper bedre)
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.utils import compute_sample_weight
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    ConfusionMatrixDisplay,
)

from load_data import load_or_build, N_BANDS


# ------------------------------------------------------------------ #
#  Feature engineering                                                 #
# ------------------------------------------------------------------ #

def pool_windows(X: np.ndarray) -> np.ndarray:
    """
    Reduser (n_files, n_windows, n_features) → (n_files, n_features * 3).

    For hvert frekvensbånd beregnes tre statistikker på tvers av vinduene:
      - mean : gjennomsnittlig energi  (stabil signaturprofil)
      - std  : temporal variasjon      (modus-spesifikk aktivitetsmønster)
      - max  : toppenergi i båndet     (sterkeste sendemoment)
    """
    mean = X.mean(axis=1)
    std  = X.std(axis=1)
    peak = X.max(axis=1)
    return np.concatenate([mean, std, peak], axis=1)


def feature_names_v2(n_features: int) -> list[str]:
    stats = ["mean", "std", "max"]
    names = []
    for stat in stats:
        for i in range(n_features):
            band = "L" if i < n_features // 2 else "H"
            idx  = i if i < n_features // 2 else i - n_features // 2
            names.append(f"{stat}_{band}_band{idx:02d}")
    return names


# ------------------------------------------------------------------ #
#  Kjøringer                                                           #
# ------------------------------------------------------------------ #

def run_dummy(X: np.ndarray, y: np.ndarray) -> None:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    print("\n--- DUMMY (Most Frequent) ---")
    print(f"Test accuracy : {dummy.score(X_test, y_test):.4f}")
    print(f"Test macro-F1 : {f1_score(y_test, dummy.predict(X_test), average='macro', zero_division=0):.4f}")


def run_kitchen_sink(X: np.ndarray, y: np.ndarray) -> None:
    """Stor modell, ingen regularisering — viser at rå kapasitet ikke hjelper."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("mlp", MLPClassifier(hidden_layer_sizes=(256, 128), alpha=0,
                              max_iter=2000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("\n--- KITCHEN SINK (Stor & Uregulert) ---")
    print(f"Train accuracy : {pipe.score(X_train, y_train):.4f}")
    print(f"Test accuracy  : {pipe.score(X_test, y_test):.4f}")
    print(f"Test  macro-F1 : {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    disp.ax_.set_title("MLP v2 Kitchen Sink — Confusion Matrix")
    plt.tight_layout()
    plt.show()


def run_mlp_v2(X: np.ndarray, y: np.ndarray) -> tuple:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Klassevekting — kompenserer for ubalanse uten å endre datasettstruktur
    sample_weights = compute_sample_weight("balanced", y_train)

    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("mlp", MLPClassifier(
            activation="relu",
            solver="adam",
            max_iter=5000,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            learning_rate="adaptive",
            random_state=42,
        )),
    ])

    # Lite søkerom — med 181 treningsfiler er store modeller bortkastet
    param_grid = {
        "mlp__hidden_layer_sizes": [(32,), (64,), (32, 16), (64, 32)],
        "mlp__alpha": [0.05, 0.1, 0.5, 1.0],
        "mlp__learning_rate_init": [0.001, 0.0005],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",   # macro-F1, ikke accuracy
        n_jobs=-1,
        verbose=1,
    )

    # sample_weight sendes gjennom Pipeline til MLPClassifier-steget
    grid.fit(X_train, y_train, mlp__sample_weight=sample_weights)

    print("\n--- MLP v2 ---")
    print("Beste parametere:", grid.best_params_)
    print(f"Beste CV macro-F1: {grid.best_score_:.4f}")

    best = grid.best_estimator_
    y_pred = best.predict(X_test)

    print(f"\nTrain accuracy : {best.score(X_train, y_train):.4f}")
    print(f"Test  accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Test  macro-F1 : {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, y_pred, zero_division=0))

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    disp.ax_.set_title("MLP v2 — Confusion Matrix")
    plt.tight_layout()
    plt.show()

    return best, grid


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    data_dir = ".DroneRF"

    # Last CNN-data (vindubasert) — samme features som CNN bruker
    X_3d, y, label_map, _ = load_or_build(data_dir, mode="cnn")

    print(f"Lastet: {X_3d.shape}")          # (n_files, n_windows, n_bands*2)
    print(f"Klassefordeling: {np.bincount(y)}")

    # Pool 300 vinduer → flat feature-vektor (192 features per fil)
    X = pool_windows(X_3d)
    print(f"Poolede features: {X.shape}")   # (n_files, n_bands*2 * 3)

    run_dummy(X, y)
    run_kitchen_sink(X, y)
    model, grid = run_mlp_v2(X, y)
