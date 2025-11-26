#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pipeline YOLO + CNN pour la détection et la classification de panneaux.

- YOLO (models/best.pt) : détecte les panneaux (1 classe "sign")
- CNN  (models/My_GTSRB_5classes_balanced.h5) : classe 5 panneaux :
    0 -> Speed limit 20 km/h  (00000)
    1 -> Speed limit 50 km/h  (00002)
    2 -> Stop                 (00014)
    3 -> Turn right           (00033)
    4 -> Turn left            (00034)

Si la confiance max du CNN < SEUIL_UNKNOWN -> label = "Inconnu".
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from PIL import Image

# ==============================
# CONFIG GLOBALE
# ==============================

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))

YOLO_WEIGHTS = os.path.join(PROJECT_ROOT, "models",  "best.pt")
CNN_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "My_GTSRB_5classes_balanced.h5")

# mapping index CNN -> label humain
IDX_TO_LABEL = {
    0: "Speed limit 20 km/h",   # 00000
    1: "Speed limit 50 km/h",   # 00002
    2: "Stop",                  # 00014
    3: "Turn right",            # 00033
    4: "Turn left",             # 00034
}

# en dessous de ce seuil, on considère que le CNN n'est pas sûr -> "Inconnu"
SEUIL_UNKNOWN = 0.8

# tailles pour le CNN
CNN_INPUT_SIZE = (32, 32)


# ==============================
# UTILITAIRES CNN
# ==============================

def load_cnn_model():
    """Charge le modèle CNN une seule fois."""
    if not os.path.exists(CNN_WEIGHTS):
        raise FileNotFoundError(f"[ERREUR] CNN introuvable : {CNN_WEIGHTS}")
    print("[INFO] Chargement CNN depuis", CNN_WEIGHTS)
    model = tf.keras.models.load_model(CNN_WEIGHTS)
    print("[INFO] CNN chargé.")
    return model


def preprocess_for_cnn(crop_bgr):
    """
    Prend un crop BGR (np.array HxWx3 uint8) et le prépare pour le CNN :
    - convert BGR->RGB
    - resize 32x32
    - normalisation [0,1]
    - ajout batch dimension (1,32,32,3)
    """
    # BGR -> RGB
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    # PIL pour resize propre
    img = Image.fromarray(crop_rgb)
    img = img.resize(CNN_INPUT_SIZE)
    x = np.array(img, dtype=np.float32) / 255.0
    x = x.reshape((1, CNN_INPUT_SIZE[0], CNN_INPUT_SIZE[1], 3))
    return x


def classify_sign(cnn_model, crop_bgr, seuil_unknown=SEUIL_UNKNOWN):
    """
    Classe un crop d'image avec le CNN.
    Retourne (label, idx, conf, proba_vector, is_unknown).
    """
    x = preprocess_for_cnn(crop_bgr)
    proba = cnn_model.predict(x, verbose=0)[0]  # vecteur de taille 5
    idx = int(np.argmax(proba))
    conf = float(proba[idx])
    label = IDX_TO_LABEL.get(idx, f"Classe {idx}")

    is_unknown = conf < seuil_unknown
    if is_unknown:
        label_final = "Inconnu"
    else:
        label_final = label

    return label_final, idx, conf, proba, is_unknown


# ==============================
# UTILITAIRES YOLO + DESSIN
# ==============================

def load_yolo_model():
    """Charge le modèle YOLO."""
    if not os.path.exists(YOLO_WEIGHTS):
        raise FileNotFoundError(f"[ERREUR] YOLO introuvable : {YOLO_WEIGHTS}")
    print("[INFO] Chargement YOLO depuis", YOLO_WEIGHTS)
    model = YOLO(YOLO_WEIGHTS)
    print("[INFO] YOLO chargé.")
    return model


def draw_detection(frame_bgr, x1, y1, x2, y2, label_text, color):
    """Dessine un rectangle + label sur l'image."""
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame_bgr,
        label_text,
        (x1, max(0, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
        cv2.LINE_AA,
    )


# ==============================
# PIPELINE SUR UNE IMAGE
# ==============================

def run_pipeline_on_image(image_path, output_path, yolo_model, cnn_model, conf_yolo=0.5):
    """
    YOLO + CNN sur une image :
      - YOLO détecte les panneaux
      - chaque crop -> CNN
      - si conf < seuil -> "Inconnu"
      - image annotée sauvegardée dans output_path
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image introuvable : {image_path}")

    print("[INFO] Traitement image :", image_path)

    # YOLO prend directement le chemin
    results = yolo_model(image_path, conf=conf_yolo, verbose=False)
    result = results[0]

    # frame BGR récupéré de YOLO
    frame_bgr = result.orig_img.copy()

    if result.boxes is None or len(result.boxes) == 0:
        print("[INFO] Aucun panneau détecté par YOLO.")
    else:
        print(f"[INFO] {len(result.boxes)} détection(s) par YOLO.")
        for box in result.boxes:
            # xyxy est un tensor [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # clipping simple
            h, w, _ = frame_bgr.shape
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w - 1))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            label_final, idx, conf_cnn, proba, is_unknown = classify_sign(cnn_model, crop)

            print(
                f"  - bbox ({x1},{y1},{x2},{y2}) -> "
                f"classe {idx} ({IDX_TO_LABEL.get(idx,'?')}) "
                f"conf={conf_cnn:.3f} -> label_final='{label_final}'"
            )

            if is_unknown:
                color = (0, 0, 255)   # rouge
            else:
                color = (0, 255, 0)   # vert

            text = f"{label_final} ({conf_cnn:.2f})"
            draw_detection(frame_bgr, x1, y1, x2, y2, text, color)

    # sauvegarde
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame_bgr)
    print("[INFO] Image annotée sauvegardée dans", output_path)


# ==============================
# PIPELINE SUR UNE VIDÉO
# ==============================

def run_pipeline_on_video(video_path, output_path, yolo_model, cnn_model, conf_yolo=0.5):
    """
    YOLO + CNN sur une vidéo :
      - pour chaque frame : YOLO -> bbox
      - chaque crop -> CNN
      - affiche "Inconnu" si conf < seuil
      - écrit aussi le temps (en secondes) dans les logs
      - enregistre une vidéo annotée dans output_path
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Vidéo introuvable : {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la vidéo : {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # fallback
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("[INFO] Traitement vidéo :", video_path)
    print(f"[INFO] fps={fps:.2f}, size={width}x{height}")

    frame_idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        t_sec = frame_idx / fps

        # YOLO sur le frame (array directement)
        results = yolo_model(frame_bgr, conf=conf_yolo, verbose=False)
        result = results[0]

        if result.boxes is not None and len(result.boxes) > 0:
            print(f"[{t_sec:6.2f}s] {len(result.boxes)} panneau(x) détecté(s)")
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                h, w, _ = frame_bgr.shape
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w - 1))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h - 1))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame_bgr[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                label_final, idx, conf_cnn, proba, is_unknown = classify_sign(cnn_model, crop)

                if is_unknown:
                    color = (0, 0, 255)  # rouge
                else:
                    color = (0, 255, 0)  # vert

                text = f"{label_final} ({conf_cnn:.2f})"
                draw_detection(frame_bgr, x1, y1, x2, y2, text, color)

                print(
                    f"    -> [{t_sec:6.2f}s] bbox ({x1},{y1},{x2},{y2}) "
                    f"classe {idx} ({IDX_TO_LABEL.get(idx, '?')}) "
                    f"conf={conf_cnn:.3f} -> '{label_final}'"
                )

        out.write(frame_bgr)
        frame_idx += 1

    cap.release()
    out.release()
    print("[INFO] Vidéo annotée sauvegardée dans", output_path)


# ==============================
# MAIN (à adapter)
# ==============================

def main():
    # --- chargement des modèles ---
    yolo_model = load_yolo_model()
    cnn_model = load_cnn_model()

    # === EXEMPLE 1 : une image ===
    image_path = os.path.join(PROJECT_ROOT, "assets", "perception", "example_sign_detection.jpg")
    output_image_path = os.path.join(PROJECT_ROOT, "assets", "perception", "output_cnn_yolo_image.jpg")

    if os.path.exists(image_path):
        run_pipeline_on_image(image_path, output_image_path, yolo_model, cnn_model, conf_yolo=0.5)
    else:
        print("[WARN] Image d'exemple introuvable :", image_path)

    # === EXEMPLE 2 : une vidéo ===
    video_path = os.path.join(PROJECT_ROOT, "assets", "perception", "example_sign_detection.mp4")
    output_video_path = os.path.join(PROJECT_ROOT, "assets", "perception", "output_cnn_yolo_video.mp4")

    if os.path.exists(video_path):
        run_pipeline_on_video(video_path, output_video_path, yolo_model, cnn_model, conf_yolo=0.5)
    else:
        print("[WARN] Vidéo d'exemple introuvable :", video_path)


if __name__ == "__main__":
    main()
