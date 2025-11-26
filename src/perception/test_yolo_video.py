#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ultralytics import YOLO
import os
import sys

WEIGHTS = "runs/detect/train3/weights/best.pt"  # ou train_gtsdb/weights/best.pt
VIDEO_PATH = "video2.mp4"  # mets ta vidéo à la racine du projet

def main():
    if not os.path.exists(WEIGHTS):
        print("[ERREUR] Poids YOLO introuvables :", WEIGHTS)
        sys.exit(1)
    if not os.path.exists(VIDEO_PATH):
        print("[ERREUR] Vidéo introuvable :", VIDEO_PATH)
        sys.exit(1)

    model = YOLO(WEIGHTS)

    print("Vidéo test :", VIDEO_PATH)
    results = model(
        source=VIDEO_PATH,
        save=True,
        conf=0.5,
        imgsz=450
    )
    print(results)
    print("[INFO] Vidéo annotée dans runs/detect/predict*/")

if __name__ == "__main__":
    main()