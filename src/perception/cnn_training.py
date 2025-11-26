# src/cnn_training.py

import os
from collections import Counter

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization,
    Flatten, Dense, Dropout
)

# ==========================
# 1. CHEMINS & PARAMÈTRES
# ==========================

SOURCE_ROOT = "GTSRB-2/Final_Training/Images"  # dossier original GTSRB
SELECTED_CLASSES = ["00000", "00002", "00014", "00033", "00034"]  # 20, 50, Stop, Right, Left

IMAGE_SIZE = (32, 32)
BATCH_SIZE = 32
EPOCHS = 15
MODEL_OUT = "models/My_GTSRB_5classes_balanced.h5"


def build_generators():
    """Crée les générateurs train/val et renvoie aussi le mapping idx->class."""
    print("[INFO] Sous-dossiers disponibles dans", SOURCE_ROOT)
    print(os.listdir(SOURCE_ROOT))
    print("[INFO] On ne retiendra que :", SELECTED_CLASSES)

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        validation_split=0.2
    )

    val_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        SOURCE_ROOT,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        classes=SELECTED_CLASSES,
        class_mode="categorical",
        subset="training",
        shuffle=True,
        seed=42
    )

    val_generator = val_datagen.flow_from_directory(
        SOURCE_ROOT,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        classes=SELECTED_CLASSES,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=42
    )

    print("\n[INFO] Mapping classe -> index :")
    print(train_generator.class_indices)

    idx_to_class = {v: k for k, v in train_generator.class_indices.items()}
    print("[INFO] idx_to_class :", idx_to_class)

    return train_generator, val_generator, idx_to_class


def compute_class_weights(train_generator):
    """Calcule les class_weights pour équilibrer l'entraînement."""
    y_classes = train_generator.classes
    counts = Counter(y_classes)
    print("\n[INFO] Nombre d'images par index de classe (train) :", counts)

    n_total = sum(counts.values())
    n_classes = len(counts)

    class_weights = {
        cls_idx: n_total / (n_classes * n)
        for cls_idx, n in counts.items()
    }

    print("\n[INFO] class_weights calculés :")
    for k, v in class_weights.items():
        print(f"  index {k} -> {v:.3f}")

    return class_weights


def build_cnn(num_classes):
    """Construit le CNN (archi Mercury)."""
    model = Sequential()

    # Bloc 1
    model.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    # Bloc 2
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    # Bloc 3
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(BatchNormalization())

    # Denses
    model.add(Flatten())
    model.add(Dense(512, activation='relu')); model.add(Dropout(0.2))
    model.add(Dense(256, activation='relu')); model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu')); model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'));  model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'));  model.add(Dropout(0.2))
    model.add(Dense(16, activation='relu'));  model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()
    return model


def train_cnn():
    """Pipeline complet : data → model → entraînement → sauvegarde."""
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)

    train_gen, val_gen, idx_to_class = build_generators()
    class_weights = compute_class_weights(train_gen)

    num_classes = len(train_gen.class_indices)
    model = build_cnn(num_classes)

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        class_weight=class_weights
    )

    model.save(MODEL_OUT)
    print(f"\n[INFO] Modèle équilibré sauvegardé dans {MODEL_OUT}")
    return model, history, idx_to_class


if __name__ == "__main__":
    train_cnn()