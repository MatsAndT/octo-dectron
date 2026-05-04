import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    sns.pairplot(df, hue="label")
    plt.show()



if __name__ == "__main__":

    data_dir = "AR drone"

    X, y, mode_map, feature_names = load_or_build(data_dir, mode="mlp")

    print_dataset_info(
        X,
        y,
        label_map=mode_map,
        feature_names=feature_names
    )