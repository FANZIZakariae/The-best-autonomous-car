#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ultralytics import YOLO

def main():
    # Charger modèle YOLO pré-entraîné
    model = YOLO("yolov8n.pt")

    # Entraîner sur ton dataset GTSDB
    model.train(
        data="gtsdb.yaml",
        epochs=10,
        imgsz=640,
        task="detect",
        name="train_gtsdb"   # runs/detect/train_gtsdb
    )

if __name__ == "__main__":
    main()