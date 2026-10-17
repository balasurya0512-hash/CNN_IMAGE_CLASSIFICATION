"""Train an RNN classifier for cat and dog images."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


IMAGE_SIZE = (96, 96)
CHANNELS = 3
CLASS_NAMES = ["cat", "dog"]


def build_model() -> tf.keras.Model:
    """Build an image-to-sequence bidirectional LSTM classifier."""
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, CHANNELS))
    x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)
    x = tf.keras.layers.RandomFlip("horizontal")(x)
    x = tf.keras.layers.RandomRotation(0.08)(x)
    x = tf.keras.layers.Reshape((IMAGE_SIZE[0], IMAGE_SIZE[1] * CHANNELS))(x)

    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs, name="cat_dog_row_rnn")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def make_datasets(
    data_dir: Path, batch_size: int, seed: int
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Create deterministic training and validation datasets from class folders."""
    common_args = {
        "directory": data_dir,
        "labels": "inferred",
        "label_mode": "int",
        "class_names": CLASS_NAMES,
        "image_size": IMAGE_SIZE,
        "batch_size": batch_size,
        "validation_split": 0.2,
        "seed": seed,
    }
    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training", shuffle=True, **common_args
    )
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation", shuffle=False, **common_args
    )
    autotune = tf.data.AUTOTUNE
    return train_ds.prefetch(autotune), validation_ds.prefetch(autotune)


def predict_label(probability: float) -> tuple[str, float]:
    """Map a sigmoid probability to the correct cat/dog label and confidence."""
    if probability >= 0.5:
        return "dog", float(probability)
    return "cat", float(1.0 - probability)


def predict_image(model: tf.keras.Model, image_path: Path) -> tuple[str, float]:
    """Return the predicted class and its probability for one image."""
    image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(image)
    probability = float(model.predict(image_array[None, ...], verbose=0)[0][0])
    label, confidence = predict_label(probability)
    return label, confidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "Data" / "Train_Data",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predict", type=Path, nargs="*", default=[])
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.data_dir}")
    for class_name in CLASS_NAMES:
        if not (args.data_dir / class_name).is_dir():
            raise FileNotFoundError(f"Missing class directory: {args.data_dir / class_name}")

    tf.keras.utils.set_random_seed(args.seed)
    train_ds, validation_ds = make_datasets(args.data_dir, args.batch_size, args.seed)
    model = build_model()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "cat_dog_rnn.keras", monitor="val_accuracy", save_best_only=True
        ),
    ]
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=args.epochs,
        callbacks=callbacks,
    )
    loss, accuracy = model.evaluate(validation_ds, verbose=0)
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {accuracy:.2%}")

    for image_path in args.predict:
        label, confidence = predict_image(model, image_path)
        print(f"{image_path}: {label} ({confidence:.2%})")


if __name__ == "__main__":
    main()