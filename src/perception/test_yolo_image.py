#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import os
import sys

WEIGHTS = "runs/detect/train3/weights/best.pt"  # ou train_gtsdb/weights/best.pt
IMG_PATH = "gtsdb_yolo/images/val/00040.jpg"    # adapte

def main():
    if not os.path.exists(WEIGHTS):
        print("[ERREUR] Poids YOLO introuvables :", WEIGHTS)
        sys.exit(1)
    if not os.path.exists(IMG_PATH):
        print("[ERREUR] Image introuvable :", IMG_PATH)
        sys.exit(1)

    model = YOLO(WEIGHTS)

    print("Image test :", IMG_PATH)
    results = model(
        source=IMG_PATH,
        save=True,
        conf=0.5
    )
    print(results)
    print("[INFO] Image annotée dans runs/detect/predict/")

if __name__ == "__main__":
    main()