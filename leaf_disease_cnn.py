import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "Computer vision"
MODEL_PATH = ROOT_DIR / "plant_disease_cnn.keras"
CLASS_NAMES_PATH = ROOT_DIR / "class_names.txt"
CLASS_LABELS_PATH = ROOT_DIR / "class_labels.json"
DEFAULT_IMAGE_SIZE = (128, 128)
DEFAULT_BATCH_SIZE = 32
DEFAULT_EPOCHS = 10


def get_class_names(data_dir=None):
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    if not directory.exists():
        raise FileNotFoundError(f"Dataset directory not found: {directory}")
    return sorted([item.name for item in directory.iterdir() if item.is_dir()])


def build_cnn_model(input_shape=(128, 128, 3), num_classes=None):
    if num_classes is None:
        num_classes = len(get_class_names())

    model = models.Sequential([
        layers.Rescaling(1.0 / 255, input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def load_image_dataset(data_dir=None, validation_split=0.30, image_size=(128, 128), batch_size=32, seed=123):
    directory = Path(data_dir) if data_dir is not None else DATA_DIR
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        str(directory),
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="int",
    )

    remaining_dataset = tf.keras.utils.image_dataset_from_directory(
        str(directory),
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=image_size,
        batch_size=None,
        label_mode="int",
    )

    class_names = train_dataset.class_names
    total_images = tf.data.experimental.cardinality(remaining_dataset).numpy()

    remaining_dataset = remaining_dataset.shuffle(
        total_images,
        seed=seed,
        reshuffle_each_iteration=False,
    )

    validation_count = total_images // 2
    validation_dataset = remaining_dataset.take(validation_count).batch(batch_size)
    test_dataset = remaining_dataset.skip(validation_count).batch(batch_size)

    autotune = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(autotune)
    validation_dataset = validation_dataset.prefetch(autotune)
    test_dataset = test_dataset.prefetch(autotune)

    return train_dataset, validation_dataset, test_dataset, class_names


def save_class_names(class_names, output_path=CLASS_NAMES_PATH):
    output = Path(output_path)
    with output.open("w", encoding="utf-8") as file:
        for name in class_names:
            file.write(f"{name}\n")
    return output


def save_class_labels(class_names, output_path=CLASS_LABELS_PATH):
    output = Path(output_path)
    with output.open("w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)
    return output


def train_model(data_dir=None, epochs=DEFAULT_EPOCHS, batch_size=DEFAULT_BATCH_SIZE, image_size=DEFAULT_IMAGE_SIZE, seed=123):
    train_dataset, validation_dataset, test_dataset, class_names = load_image_dataset(
        data_dir=data_dir,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )

    model = build_cnn_model(input_shape=(image_size[0], image_size[1], 3), num_classes=len(class_names))
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    print()
    print("Classes:")
    print(class_names)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
    )

    results = model.evaluate(test_dataset, return_dict=True)
    print()
    print("====================================")
    print("FINAL TEST RESULTS")
    print("====================================")
    print(f"Test Loss     : {results['loss']:.4f}")
    print(f"Test Accuracy : {results['accuracy']:.4f}")

    model.save(MODEL_PATH)
    save_class_names(class_names, CLASS_NAMES_PATH)
    save_class_labels(class_names, CLASS_LABELS_PATH)

    print()
    print("====================================")
    print("MODEL SAVED")
    print("====================================")
    print(MODEL_PATH)
    print(CLASS_NAMES_PATH)
    return model, history, results, class_names


def predict_single_image(image_path, model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH, image_size=DEFAULT_IMAGE_SIZE):
    image_source = Path(image_path)
    if not image_source.exists():
        raise FileNotFoundError(f"Image not found: {image_source}")

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    class_names_path = Path(class_names_path)
    if not class_names_path.exists():
        derived_names = get_class_names()
        save_class_names(derived_names, class_names_path)

    with class_names_path.open("r", encoding="utf-8") as file:
        class_names = [line.strip() for line in file if line.strip()]

    model = tf.keras.models.load_model(str(model_path))
    image = tf.keras.utils.load_img(str(image_source), target_size=image_size)
    image_array = tf.keras.utils.img_to_array(image).astype("float32") / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)[0]
    predicted_index = int(np.argmax(prediction))
    predicted_label = class_names[predicted_index]
    predicted_confidence = float(prediction[predicted_index])
    return predicted_label, predicted_confidence, class_names


def _parse_args():
    parser = argparse.ArgumentParser(description="Leaf disease image classification")
    parser.add_argument("--train", action="store_true", help="Train the CNN model")
    parser.add_argument("--predict", type=str, help="Path to an image to predict")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Training batch size")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Dataset directory")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.train:
        train_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    elif args.predict:
        label, confidence, class_names = predict_single_image(args.predict)
        print()
        print("==============================")
        print("PLANT DISEASE PREDICTION")
        print("==============================")
        print(f"Disease: {label}")
        print(f"Confidence: {confidence * 100:.2f}%")
        print(f"Classes: {class_names}")
    else:
        print("No action selected. Use --train or --predict <image_path>.")
        raise SystemExit(0)
