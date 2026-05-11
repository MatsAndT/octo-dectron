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