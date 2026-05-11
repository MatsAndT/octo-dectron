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