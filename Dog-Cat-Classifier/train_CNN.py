"""Train and predict a Cat vs Dog CNN classifier using MobileNetV2."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
CHANNELS = 3

CLASS_NAMES = ["cat", "dog"]

MODEL_PATH = Path("cat_dog_cnn.keras")


def validate_training_data(data_dir: Path) -> None:
    """Ensure there are enough labeled samples for training and validation."""
    min_per_class = 4

    for class_name in CLASS_NAMES:
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class directory:\n{class_dir}"
            )

        image_files = list(class_dir.glob("*"))
        image_files = [p for p in image_files if p.suffix.lower() in {".bmp", ".gif", ".jpeg", ".jpg", ".png"}]

        if len(image_files) < min_per_class:
            raise ValueError(
                f"Not enough images in {class_dir}. "
                f"Found {len(image_files)} files, but at least {min_per_class} images per class are required "
                f"for a reliable split with validation. "
                "Add more images to the cat/ and dog/ folders and retry."
            )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model() -> tf.keras.Model:
    """
    Build a MobileNetV2 transfer-learning model.

    MobileNetV2 is already trained on ImageNet.
    We use it as the feature extractor and add
    our own Cat/Dog classification layers.
    """

    # --------------------------------------------------------
    # Load pretrained MobileNetV2
    # --------------------------------------------------------

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, CHANNELS),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )

    # Initially freeze MobileNetV2.
    # Its pretrained weights will not be changed.
    base_model.trainable = False

    # --------------------------------------------------------
    # Input layer
    # --------------------------------------------------------

    inputs = tf.keras.Input(
        shape=(*IMAGE_SIZE, CHANNELS),
        name="image"
    )

    # --------------------------------------------------------
    # MobileNetV2 preprocessing
    #
    # IMPORTANT:
    # We do preprocessing ONLY here.
    # Do NOT preprocess the dataset separately.
    # --------------------------------------------------------

    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)

    # --------------------------------------------------------
    # MobileNetV2 feature extraction
    # --------------------------------------------------------

    x = base_model(x, training=False)

    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    x = tf.keras.layers.Dropout(
        0.3,
        name="dropout"
    )(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu",
        name="dense_128"
    )(x)

    # One output neuron because this is binary classification.
    #
    # sigmoid output:
    #
    # close to 0 → cat
    # close to 1 → dog
    #

    outputs = tf.keras.layers.Dense(
        1,
        activation="sigmoid",
        name="dog_probability"
    )(x)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = tf.keras.Model(
        inputs=inputs,
        outputs=outputs,
        name="cat_dog_mobilenetv2"
    )

    # --------------------------------------------------------
    # Compile model
    # --------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model


# ============================================================
# FINE-TUNING
# ============================================================

def fine_tune_model(model: tf.keras.Model) -> None:
    """
    Unfreeze the last 20 layers of MobileNetV2
    for fine-tuning.
    """

    # Find MobileNetV2 inside our model.
    backbone = model.get_layer(
        "mobilenetv2_1.00_224"
    )

    # Allow MobileNetV2 layers to train.
    backbone.trainable = True

    # Number of layers to keep frozen.
    fine_tune_at = max(
        len(backbone.layers) - 20,
        0
    )

    # Freeze all layers before the last 20.
    for layer in backbone.layers[:fine_tune_at]:
        layer.trainable = False

    # --------------------------------------------------------
    # Recompile with a much smaller learning rate.
    # --------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-5
        ),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )


# ============================================================
# CREATE DATASETS
# ============================================================

def make_datasets(
    data_dir: Path,
    batch_size: int,
    seed: int,
) -> tuple[
    tf.data.Dataset,
    tf.data.Dataset,
]:
    """
    Create training and validation datasets.

    Expected folder structure:

    Train_Data/
    ├── cat/
    │   ├── cat1.jpg
    │   ├── cat2.jpg
    │   └── ...
    │
    └── dog/
        ├── dog1.jpg
        ├── dog2.jpg
        └── ...

    IMPORTANT:
    We DO NOT call preprocess_input() here.

    The model itself performs preprocessing.
    """

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

    # --------------------------------------------------------
    # Training dataset
    # --------------------------------------------------------

    train_ds = tf.keras.utils.image_dataset_from_directory(
        subset="training",
        shuffle=True,
        **common_args,
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    validation_ds = tf.keras.utils.image_dataset_from_directory(
        subset="validation",
        shuffle=False,
        **common_args,
    )

    # --------------------------------------------------------
    # Prefetch
    # --------------------------------------------------------

    autotune = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(
        autotune
    )

    validation_ds = validation_ds.prefetch(
        autotune
    )

    return train_ds, validation_ds


# ============================================================
# PREDICTION LABEL
# ============================================================

def predict_label(
    probability: float,
) -> tuple[str, float]:
    """
    Convert sigmoid probability into Cat/Dog label.

    probability >= 0.5
        → dog

    probability < 0.5
        → cat
    """

    if probability >= 0.5:

        return (
            "dog",
            probability
        )

    return (
        "cat",
        1.0 - probability
    )


# ============================================================
# PREDICT ONE IMAGE
# ============================================================

def predict_image(
    model: tf.keras.Model,
    image_path: Path,
) -> tuple[str, float]:
    """
    Predict whether an image is a cat or dog.
    """

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = tf.keras.utils.load_img(
        image_path,
        target_size=IMAGE_SIZE,
    )

    # --------------------------------------------------------
    # Convert image to NumPy array
    # --------------------------------------------------------

    image_array = tf.keras.utils.img_to_array(
        image
    )

    # --------------------------------------------------------
    # Add batch dimension
    #
    # Before:
    #
    # (224, 224, 3)
    #
    # After:
    #
    # (1, 224, 224, 3)
    # --------------------------------------------------------

    image_array = image_array[None, ...]

    # --------------------------------------------------------
    # IMPORTANT
    #
    # DO NOT call preprocess_input() here.
    #
    # The model already performs:
    #
    # preprocess_input()
    #
    # inside build_model().
    # --------------------------------------------------------

    probability = float(
        model.predict(
            image_array,
            verbose=0
        )[0][0]
    )

    # --------------------------------------------------------
    # Convert probability to label
    # --------------------------------------------------------

    label, confidence = predict_label(
        probability
    )

    return label, confidence


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(
    data_dir: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    fine_tune_epochs: int,
) -> tf.keras.Model:

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_ds, validation_ds = make_datasets(
        data_dir=data_dir,
        batch_size=batch_size,
        seed=seed,
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_model()

    print("\n================ MODEL SUMMARY ================\n")

    model.summary()

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    callbacks = [

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
        ),
    ]

    # ========================================================
    # PHASE 1
    # Transfer Learning
    # ========================================================

    print(
        "\n================================================"
    )
    print(
        "PHASE 1: TRANSFER LEARNING"
    )
    print(
        "================================================\n"
    )

    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    # ========================================================
    # PHASE 2
    # Fine-Tuning
    # ========================================================

    if fine_tune_epochs > 0:

        print(
            "\n================================================"
        )
        print(
            "PHASE 2: FINE-TUNING"
        )
        print(
            "================================================\n"
        )

        fine_tune_model(model)

        model.fit(
            train_ds,
            validation_data=validation_ds,
            epochs=fine_tune_epochs,
            callbacks=callbacks,
        )

    # --------------------------------------------------------
    # Load best saved model
    # --------------------------------------------------------

    if MODEL_PATH.exists():

        model = tf.keras.models.load_model(
            MODEL_PATH
        )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    loss, accuracy = model.evaluate(
        validation_ds,
        verbose=0,
    )

    print(
        f"\nValidation loss: {loss:.4f}"
    )

    print(
        f"Validation accuracy: {accuracy:.2%}"
    )

    print(
        f"\nModel saved as: {MODEL_PATH}"
    )

    return model


# ============================================================
# PREDICT MODE
# ============================================================

def predict_images(
    model_path: Path,
    image_paths: list[Path],
) -> None:

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train the model first."
        )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print(
        f"Loading model: {model_path}"
    )

    model = tf.keras.models.load_model(
        model_path
    )

    # --------------------------------------------------------
    # Predict every image
    # --------------------------------------------------------

    for image_path in image_paths:

        if not image_path.exists():

            print(
                f"\nFile not found: {image_path}"
            )

            continue

        label, confidence = predict_image(
            model,
            image_path,
        )

        print(
            f"\n{image_path}"
        )

        print(
            f"Prediction : {label}"
        )

        print(
            f"Confidence : {confidence:.2%}"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    # --------------------------------------------------------
    # Dataset location
    # --------------------------------------------------------

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=(
            Path(__file__).parent
            / "Data"
            / "Train_Data"
        ),
    )

    # --------------------------------------------------------
    # Training epochs
    # --------------------------------------------------------

    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
    )

    # --------------------------------------------------------
    # Batch size
    # --------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    # --------------------------------------------------------
    # Random seed
    # --------------------------------------------------------

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    # --------------------------------------------------------
    # Fine-tuning epochs
    # --------------------------------------------------------

    parser.add_argument(
        "--fine-tune-epochs",
        type=int,
        default=5,
    )

    # --------------------------------------------------------
    # Prediction images
    # --------------------------------------------------------

    parser.add_argument(
        "--predict",
        type=Path,
        nargs="*",
        default=[],
    )

    # --------------------------------------------------------
    # Parse arguments
    # --------------------------------------------------------

    args = parser.parse_args()

    # ========================================================
    # PREDICTION ONLY MODE
    # ========================================================

    if args.predict:

        predict_images(
            model_path=MODEL_PATH,
            image_paths=args.predict,
        )

        return

    # ========================================================
    # TRAINING MODE
    # ========================================================

    # --------------------------------------------------------
    # Check dataset
    # --------------------------------------------------------

    if not args.data_dir.exists():

        raise FileNotFoundError(
            f"Dataset directory does not exist:\n"
            f"{args.data_dir}"
        )

    validate_training_data(args.data_dir)

    # --------------------------------------------------------
    # Set random seed
    # --------------------------------------------------------

    tf.keras.utils.set_random_seed(
        args.seed
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        fine_tune_epochs=args.fine_tune_epochs,
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()