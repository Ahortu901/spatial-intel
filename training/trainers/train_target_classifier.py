"""
Model 1 — Micro-Doppler CNN Target Classifier
Classifies radar targets as: person / vehicle / drone / empty / animal
Input:  2D STFT spectrogram [33 x 20 x 1]
Output: class probabilities
Architecture: MobileNetV2-style depthwise separable CNN — fast on CM5
"""

import numpy as np
import os
import sys
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

TARGET_LABEL_NAMES = [
    "empty", "person", "vehicle_car",
    "vehicle_truck", "drone_quad", "drone_fixedwing", "animal"
]
TARGET_LABEL_TO_IDX = {l: i for i, l in enumerate(TARGET_LABEL_NAMES)}
NUM_TARGET_CLASSES = len(TARGET_LABEL_NAMES)


def build_cnn_classifier(input_shape=(33, 20, 1), num_classes=NUM_TARGET_CLASSES):
    """
    Lightweight CNN for spectrogram classification.
    MobileNet-style depthwise separable convolutions — 50x fewer parameters
    than standard CNN, but similar accuracy. Fits CM5 memory budget.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, Model

    inp = tf.keras.Input(shape=input_shape)

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)

    # Depthwise separable blocks
    def dw_block(x, filters, stride=1):
        x = layers.DepthwiseConv2D(3, strides=stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU(6.0)(x)
        return x

    x = dw_block(x, 64,  stride=1)
    x = dw_block(x, 128, stride=2)
    x = dw_block(x, 128, stride=1)
    x = dw_block(x, 256, stride=2)
    x = dw_block(x, 256, stride=1)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    return Model(inp, out, name="doppler_cnn")


def train_target_classifier(dataset_dir: str = "training/datasets",
                             output_dir: str = "models",
                             epochs: int = 30,
                             simulate_data: bool = False):
    import tensorflow as tf
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from training.data_collector.collect import DataCollector, load_dataset

    print("=" * 50)
    print("TRAINING: Target Classifier (Micro-Doppler CNN)")
    print("=" * 50)

    if simulate_data:
        print("Generating synthetic training data...")
        X, y = _generate_synthetic_target_data(n_per_class=300)
    else:
        X, y = load_dataset(dataset_dir, labels=TARGET_LABEL_NAMES,
                            feature="spectrogram")

    # Remap label indices to target-classifier-local indices
    y_local = np.array([TARGET_LABEL_TO_IDX.get(
        [k for k, v in TARGET_LABEL_TO_IDX.items()][int(yi) % NUM_TARGET_CLASSES],
        int(yi) % NUM_TARGET_CLASSES) for yi in y])

    # Add channel dim if needed
    if X.ndim == 3:
        X = X[..., np.newaxis]

    X = X.astype(np.float32)
    # Normalise per-sample
    X = (X - X.mean(axis=(1, 2, 3), keepdims=True)) / \
        (X.std(axis=(1, 2, 3), keepdims=True) + 1e-8)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y_local, test_size=0.2, stratify=y_local, random_state=42)

    print(f"Train: {len(X_train)}  Val: {len(X_val)}  Classes: {NUM_TARGET_CLASSES}")

    # Class weights for imbalanced datasets
    cw = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight = dict(enumerate(cw))

    model = build_cnn_classifier(
        input_shape=X.shape[1:], num_classes=NUM_TARGET_CLASSES)
    model.summary(print_fn=lambda x: print(x))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5, patience=3, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(output_dir, "target_classifier_best.h5"),
            save_best_only=True, verbose=0)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1
    )

    val_acc = max(history.history['val_accuracy'])
    print(f"\nBest validation accuracy: {val_acc:.3f}")

    # Convert to TFLite INT8
    tflite_path = _convert_to_tflite(
        model, X_train[:100],
        os.path.join(output_dir, "target_classifier.tflite"))

    # Save label map
    meta = {
        "model": "target_classifier",
        "labels": TARGET_LABEL_NAMES,
        "input_shape": list(X.shape[1:]),
        "val_accuracy": float(val_acc),
        "tflite_path": tflite_path
    }
    with open(os.path.join(output_dir, "target_classifier_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Model saved → {output_dir}/target_classifier.tflite")
    return model, tflite_path


def _generate_synthetic_target_data(n_per_class=300):
    """Fast synthetic spectrogram generation for testing."""
    from training.data_collector.collect import DataCollector, FeatureExtractor
    extractor = FeatureExtractor()
    collector = DataCollector.__new__(DataCollector)
    collector.extractor = extractor

    X, y = [], []
    for label in TARGET_LABEL_NAMES:
        idx = TARGET_LABEL_TO_IDX[label]
        for _ in range(n_per_class):
            iq = collector._simulate_iq.__func__(collector, label) \
                 if hasattr(collector._simulate_iq, '__func__') \
                 else DataCollector._DataCollector__simulate_iq(collector, label) \
                 if hasattr(DataCollector, '_DataCollector__simulate_iq') \
                 else _sim_iq(label)
            frame = extractor.extract(iq, label)
            X.append(frame.spectrogram)
            y.append(idx)
    return np.array(X), np.array(y)


def _sim_iq(label):
    """Fallback synthetic I/Q generator."""
    import time
    N, M = 256, 128
    c = 3e8
    B = 0.25e9
    f0 = 24e9
    t = time.time()
    iq = (np.random.randn(M, N) + 1j * np.random.randn(M, N)) * 0.03

    configs = {
        "empty": None,
        "person": (4.0, 0.006, 0.28),
        "vehicle_car": (8.0, 0.0, 0.0),
        "vehicle_truck": (10.0, 0.0, 0.0),
        "drone_quad": (6.0, 0.0, 0.0),
        "drone_fixedwing": (12.0, 0.0, 0.0),
        "animal": (4.0, 0.002, 0.15),
    }
    cfg = configs.get(label)
    if cfg:
        r, bv, bf = cfg
        r += bv * np.sin(2 * np.pi * bf * t)
        if "drone" in label:
            r += 0.1 * np.sin(2 * np.pi * 120 * t)
        fb = 2 * B * r / (c * 40e-6)
        t_fast = np.linspace(0, 40e-6, N)
        for chirp in range(M):
            iq[chirp] += 0.8 * np.exp(1j * (2 * np.pi * fb * t_fast + 4*np.pi*f0*r/c))
    return iq


def _convert_to_tflite(model, representative_data, output_path):
    """Convert Keras model to quantised TFLite."""
    import tensorflow as tf
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for i in range(min(100, len(representative_data))):
            yield [representative_data[i:i+1].astype(np.float32)]

    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32

    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    print(f"TFLite model: {size_kb:.1f} KB → {output_path}")
    return output_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--dataset-dir", default="training/datasets")
    p.add_argument("--output-dir", default="models")
    args = p.parse_args()
    train_target_classifier(args.dataset_dir, args.output_dir,
                            args.epochs, args.simulate)
