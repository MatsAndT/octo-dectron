import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from load_data import load_or_build

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.dummy import DummyClassifier
from sklearn.decomposition import PCA
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

def run_mlp_kitchen_sink(X, y):
    # Vi bruker alle originale features uten filtrering
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Standard oppsett: ingen regularisering, standard lag
    pipe = Pipeline([
        ("scaler", MinMaxScaler()), 
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
            max_iter=8000,
            learning_rate="adaptive",
            random_state=42
        ))
    ])

    param_grid = {
        # Prøver ulike arkitekturer, fra veldig enkle til moderat komplekse
        "mlp__hidden_layer_sizes": [
            (32, 32, 32), (64,),
            (16, 16, 16),
        ],
        
        
        # Regularisering (L2-straff): Dette er den viktigste bremsen mot overfitting
        "mlp__alpha": [0.001, 0.5, 1.0],
        
        # Læringsrate: Vi holder den lav for å sikre stabil konvergens
        "mlp__learning_rate_init": [0.001, 0.0005],
        
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

    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train_filtered, y_train)
    print(f"Baseline (tippe mest vanlig): {dummy.score(X_test_filtered, y_test):.2f}")

    return best_model, grid

if __name__ == "__main__":

    data_dir = ".DroneRF" 

    X, y, mode_map, feature_names = load_or_build(data_dir, mode="mlp")


    run_mlp_kitchen_sink(X, y)
    model, grid = run_mlp_pipeline(X, y, feature_names)