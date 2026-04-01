"""Train an improved facial emotion recognition model with augmentation and class balancing."""

import argparse
import json
import os

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator


EMOTION_LABELS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


def build_optimizer() -> tf.keras.optimizers.Optimizer:
    """Use the faster legacy Adam optimizer on Apple Silicon when available."""
    legacy_optimizers = getattr(tf.keras.optimizers, "legacy", None)
    if legacy_optimizers is not None and hasattr(legacy_optimizers, "Adam"):
        return legacy_optimizers.Adam(learning_rate=1e-3)
    return tf.keras.optimizers.Adam(learning_rate=1e-3)


def build_model(num_classes: int) -> tf.keras.Model:
    """Build a stronger CNN for emotion recognition."""
    model = models.Sequential([
        layers.Input(shape=(48, 48, 1)),

        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.30),

        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.35),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.40),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.30),
        layers.Dense(num_classes, activation="softmax"),
    ])

    model.compile(
        optimizer=build_optimizer(),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(class_indices: dict, train_dir: str) -> dict:
    """Compute inverse-frequency class weights to help underrepresented emotions."""
    class_counts = {}
    for class_name, class_id in class_indices.items():
        class_path = os.path.join(train_dir, class_name)
        count = len([name for name in os.listdir(class_path) if not name.startswith('.')])
        class_counts[class_id] = max(count, 1)

    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    return {
        class_id: total_samples / (num_classes * count)
        for class_id, count in class_counts.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an improved emotion recognition model")
    parser.add_argument("--train-dir", default="dataset/train", help="Training dataset directory")
    parser.add_argument("--val-dir", default="dataset/test", help="Validation dataset directory")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--output", default="Emomodel_improved.h5", help="Output model file")
    args = parser.parse_args()

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        zoom_range=0.15,
        shear_range=0.10,
        horizontal_flip=True,
        brightness_range=(0.8, 1.2),
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    validation_generator = val_datagen.flow_from_directory(
        args.val_dir,
        target_size=(48, 48),
        color_mode="grayscale",
        batch_size=args.batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    class_weights = compute_class_weights(train_generator.class_indices, args.train_dir)
    print("Class indices:", train_generator.class_indices)
    print("Class weights:", class_weights)

    model = build_model(num_classes=train_generator.num_classes)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        ModelCheckpoint(args.output, monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    final_loss, final_acc = model.evaluate(validation_generator, verbose=1)
    print(f"Final validation accuracy: {final_acc * 100:.2f}%")

    json_path = os.path.splitext(args.output)[0] + ".json"
    with open(json_path, "w", encoding="utf-8") as json_file:
        json_file.write(model.to_json())

    labels_path = os.path.splitext(args.output)[0] + "_labels.json"
    ordered_labels = [label for label, _ in sorted(train_generator.class_indices.items(), key=lambda item: item[1])]
    with open(labels_path, "w", encoding="utf-8") as label_file:
        json.dump(ordered_labels, label_file, indent=2)

    print(f"Saved improved model to {args.output}")
    print(f"Saved architecture to {json_path}")
    print(f"Saved labels to {labels_path}")


if __name__ == "__main__":
    main()
