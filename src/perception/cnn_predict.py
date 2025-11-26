# src/cnn_predict.py

import os
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "models/My_GTSRB_5classes_balanced.h5"

IDX_TO_LABEL = {
    0: "Speed limit 20 km/h",   # 00000
    1: "Speed limit 50 km/h",   # 00002
    2: "Stop",                  # 00014
    3: "Turn right",            # 00033
    4: "Turn left",             # 00034
}


def load_model(model_path=MODEL_PATH):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Modèle introuvable : {model_path}")
    print(f"[INFO] Chargement du modèle depuis {model_path} ...")
    model = tf.keras.models.load_model(model_path)
    print("[INFO] Modèle chargé.")
    return model


def preprocess_image(img_path: str, target_size=(32, 32)):
    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image introuvable : {img_path}")
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    x = np.array(img, dtype=np.float32) / 255.0
    x = x.reshape(1, target_size[0], target_size[1], 3)
    return x


def predict_image(img_path, model_path=MODEL_PATH):
    model = load_model(model_path)
    x = preprocess_image(img_path)

    proba = model.predict(x, verbose=0)[0]
    cls_idx = int(np.argmax(proba))
    label = IDX_TO_LABEL.get(cls_idx, f"Classe {cls_idx} (inconnue)")

    print(f"\nImage : {img_path}")
    print(f"→ Classe prédite : {cls_idx} ({label})")
    print("→ Probabilités :")
    for i, p in enumerate(proba):
        print(f"   {i} ({IDX_TO_LABEL[i]}): {p:.4f}")

    return cls_idx, proba


if __name__ == "__main__":
    # exemple :
    predict_image("stop.jpg")