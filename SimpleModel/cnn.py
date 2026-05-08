import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

from load_data import load_or_build, MAX_WINDOWS, N_BANDS


def build_model(input_shape, num_classes):
    """1-D CNN over time-windows; each step has N_BANDS*2 frequency features."""
    model = models.Sequential([
        layers.Input(shape=input_shape),           # (MAX_WINDOWS, N_BANDS*2)

        layers.Conv1D(64, 5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(128, 5, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(256, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.GlobalAveragePooling1D(),

        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),

        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def main():
    data_dir = ".DroneRF"
    X, y, label_map, _ = load_or_build(data_dir, mode="cnn")

    print("Shape:", X.shape)          # expected: (n_files, MAX_WINDOWS, N_BANDS*2)
    print("Unique labels:", np.unique(y))
    print("Class distribution:", np.bincount(y))

    num_classes = len(np.unique(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Per-sample max normalisation (keeps relative frequency structure intact)
    def normalise(arr):
        m = np.max(np.abs(arr), axis=(1, 2), keepdims=True)
        m = np.where(m == 0, 1.0, m)
        return arr / m

    X_train = normalise(X_train)
    X_test  = normalise(X_test)

    # input_shape = (MAX_WINDOWS, N_BANDS * 2)
    input_shape = (X_train.shape[1], X_train.shape[2])
    print("Input shape per sample:", input_shape)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    class_weights = dict(enumerate(class_weights))
    print("Class weights:", class_weights)

    model = build_model(input_shape, num_classes)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
    )

    def data_generator(X, y, cw, batch_size=32):
        while True:
            idx = np.random.randint(0, len(X), batch_size)
            batch_x = X[idx].copy()
            batch_y = y[idx]

            # Light augmentation: Gaussian noise + temporal shift
            batch_x += np.random.normal(0, 0.01, batch_x.shape)
            shift = np.random.randint(-10, 10)
            batch_x = np.roll(batch_x, shift, axis=1)

            sample_weights = np.array([cw[label] for label in batch_y])
            yield batch_x, batch_y, sample_weights

    print("\n--- TRAINING ---")
    history = model.fit(
        data_generator(X_train, y_train, class_weights),
        steps_per_epoch=len(X_train) // 32,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
    )

    print("\n--- SANITY CHECK ---")
    model.fit(X_train[:50], y_train[:50], epochs=50, verbose=0)
    train_acc = model.evaluate(X_train[:50], y_train[:50], verbose=0)[1]
    print(f"Overfit test accuracy (should be ~1.0): {train_acc:.3f}")

    print("\n--- TEST ---")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("\n--- REPORT ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    disp.ax_.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
