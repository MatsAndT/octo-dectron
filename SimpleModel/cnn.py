import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight

from load_data import load_or_build


# -------------------------
# 🔧 Enklere og mer robust CNN
# -------------------------
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv1D(32, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.Conv1D(64, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        # Viktig: behold struktur
        layers.GlobalAveragePooling1D(),

        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


# -------------------------
#  Enkel RF-augmentering
# -------------------------
def augment_batch(X):
    noise = np.random.normal(0, 0.01, X.shape)
    shift = np.roll(X, np.random.randint(-5, 5), axis=1)
    return X + noise + shift


def main():
    # 1. Last data
    data_dir = ".DroneRF"
    X, y, label_map, _ = load_or_build(data_dir, mode="cnn")

    print("Unike labels:", np.unique(y))
    print("Fordeling:", np.bincount(y))

    num_classes = len(np.unique(y))
    print("Class distribution:", np.bincount(y))

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -------------------------
    # 🔧 Normalisering (ikke StandardScaler)
    # -------------------------
    X_train = X_train / np.max(np.abs(X_train))
    X_test = X_test / np.max(np.abs(X_test))

    # -------------------------
    # 🔧 Reshape til CNN
    # -------------------------
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    print("Shape:", X_train.shape)

    # -------------------------
    # 🔧 Class weights (KRITISK)
    # -------------------------
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = dict(enumerate(class_weights))

    print("Class weights:", class_weights)

    # -------------------------
    # 🔧 Modell
    # -------------------------
    model = build_model((X_train.shape[1], 1), num_classes)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    # -------------------------
    # 🔧 Treningsdata + augmentering
    # -------------------------
    def data_generator(X, y, class_weights, batch_size=16):
        class_labels = np.array(sorted(class_weights.keys()), dtype=np.int64)
        class_indices = {label: np.where(y == label)[0] for label in class_labels}

        while True:
            sampled_labels = np.random.choice(class_labels, size=batch_size, replace=True)
            idx = np.array(
                [np.random.choice(class_indices[label]) for label in sampled_labels],
                dtype=np.int64,
            )
            batch_x = X[idx]
            batch_y = y[idx]

            noise = np.random.normal(0, 0.01, batch_x.shape)
            shift = np.roll(batch_x, np.random.randint(-5, 5), axis=1)
            batch_x = batch_x + noise + shift

            # legg på vekter
            sample_weights = np.array([class_weights[label] for label in batch_y])

            yield batch_x, batch_y, sample_weights

    print("\n--- TRAINING ---")

    history = model.fit(
        data_generator(X_train, y_train, class_weights),
        steps_per_epoch=len(X_train)//16,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )

    # -------------------------
    # 🔧 Sanity check (overfit liten batch)
    # -------------------------
    print("\n--- SANITY CHECK ---")
    model.fit(X_train[:50], y_train[:50], epochs=50, verbose=0)
    train_acc = model.evaluate(X_train[:50], y_train[:50], verbose=0)[1]
    print(f"Overfit test accuracy (should be ~1.0): {train_acc:.3f}")

    # -------------------------
    # 🔧 Evaluering
    # -------------------------
    print("\n--- TEST ---")

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)

    print("\n--- REPORT ---")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        cmap="Blues"
    )
    disp.ax_.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
    
