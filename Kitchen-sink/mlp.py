import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from load_data import load_or_build

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)



def run_mlp_pipeline(X, y):


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            activation="relu",
            solver="adam",
            max_iter=2000,
            random_state=42
        ))
    ])


    param_grid = {
        "mlp__hidden_layer_sizes": [(64,), (128,), (128, 64), (256, 128)],
        "mlp__alpha": [1e-5, 1e-4, 1e-3],
        "mlp__learning_rate_init": [1e-4, 1e-3],
    }

    grid = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1
    )


    grid.fit(X_train, y_train)

    print("\nBEST PARAMETERS:")
    print(grid.best_params_)
    print("\nBEST CV SCORE:", grid.best_score_)

    best_model = grid.best_estimator_


    y_train_pred = best_model.predict(X_train)

    print("\nTRAIN RESULTS")
    print("Accuracy:", accuracy_score(y_train, y_train_pred))
    print(classification_report(y_train, y_train_pred, zero_division=0))


    y_test_pred = best_model.predict(X_test)

    print("\nTEST RESULTS")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))
    print(classification_report(y_test, y_test_pred, zero_division=0))


    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred)
    plt.title("Confusion Matrix (Test)")
    plt.show()

    return best_model, grid


if __name__ == "__main__":

    data_dir = "AR drone"

    X, y, mode_map, feature_names = load_or_build(data_dir, mode="mlp")


    model, grid = run_mlp_pipeline(X, y)