#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prépare le dataset GTSDB au format YOLO (1 classe "sign").
À placer dans : src/perception/gtsdb_to_yolo.py
"""

import os
import random
from collections import defaultdict
from PIL import Image
import shutil
import glob

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# === chemins ===
# ce fichier est dans src/perception -> on remonte de 2 niveaux pour aller à la racine du projet
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(THIS_DIR))

BASE_DIR = os.path.join(PROJECT_ROOT, "archive")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "TrainIJCNN2013", "TrainIJCNN2013")
TRAIN_GT_FILE = os.path.join(TRAIN_IMG_DIR, "gt.txt")

OUT_ROOT = os.path.join(PROJECT_ROOT, "gtsdb_yolo")
TRAIN_SPLIT = 0.8
YOLO_CLASS_ID = 0  # "sign"


def parse_gt(gt_file):
    boxes_par_image = defaultdict(list)
    with open(gt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            if len(parts) != 6:
                continue
            fname = parts[0]
            x1, y1, x2, y2 = map(int, parts[1:5])
            class_id_raw = int(parts[5])
            boxes_par_image[fname].append((x1, y1, x2, y2, class_id_raw))
    return boxes_par_image


def create_yolo_dirs(out_root):
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(out_root, sub), exist_ok=True)


def convert_to_yolo(x1, y1, x2, y2, img_w, img_h):
    bx = (x1 + x2) / 2.0
    by = (y1 + y2) / 2.0
    bw = (x2 - x1)
    bh = (y2 - y1)
    return bx / img_w, by / img_h, bw / img_w, bh / img_h


def convert_gtsdb_to_yolo():
    print("[INFO] Lecture des annotations depuis", TRAIN_GT_FILE)
    boxes_par_image = parse_gt(TRAIN_GT_FILE)
    filenames = sorted(boxes_par_image.keys())
    print(f"[INFO] {len(filenames)} images annotées")

    random.seed(42)
    random.shuffle(filenames)
    n_train = int(len(filenames) * TRAIN_SPLIT)
    train_files = set(filenames[:n_train])

    print(f"[INFO] Split : {n_train} train, {len(filenames) - n_train} val")
    create_yolo_dirs(OUT_ROOT)

    for fname in tqdm(filenames, desc="Conversion GTSDB -> YOLO"):
        src_img_path = os.path.join(TRAIN_IMG_DIR, fname)
        if not os.path.isfile(src_img_path):
            print("[WARN] Image manquante :", src_img_path)
            continue

        with Image.open(src_img_path) as im:
            w, h = im.size

        yolo_lines = []
        for (x1, y1, x2, y2, class_raw) in boxes_par_image[fname]:
            x_c, y_c, w_n, h_n = convert_to_yolo(x1, y1, x2, y2, w, h)
            yolo_lines.append(
                f"{YOLO_CLASS_ID} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
            )

        if fname in train_files:
            img_out_dir = os.path.join(OUT_ROOT, "images/train")
            lbl_out_dir = os.path.join(OUT_ROOT, "labels/train")
        else:
            img_out_dir = os.path.join(OUT_ROOT, "images/val")
            lbl_out_dir = os.path.join(OUT_ROOT, "labels/val")

        stem = os.path.splitext(fname)[0]
        dst_img_path = os.path.join(img_out_dir, fname)
        dst_lbl_path = os.path.join(lbl_out_dir, stem + ".txt")

        shutil.copy2(src_img_path, dst_img_path)
        with open(dst_lbl_path, "w") as f_lbl:
            f_lbl.write("\n".join(yolo_lines))

    print("[INFO] Conversion GT terminée ->", OUT_ROOT)


def convert_ppm_to_jpg():
    for split in ["train", "val"]:
        img_dir = os.path.join(OUT_ROOT, "images", split)
        ppm_files = glob.glob(os.path.join(img_dir, "*.ppm"))
        print(f"{split} : {len(ppm_files)} fichiers .ppm à convertir")
        for ppm_path in ppm_files:
            with Image.open(ppm_path) as im:
                im = im.convert("RGB")
                jpg_path = os.path.splitext(ppm_path)[0] + ".jpg"
                im.save(jpg_path, "JPEG")
    print("[INFO] Conversion PPM -> JPG terminée.")


def create_gtsdb_yaml():
    yaml_path = os.path.join(PROJECT_ROOT, "gtsdb.yaml")
    yaml_content = """path: ./gtsdb_yolo
train: images/train
val: images/val

nc: 1
names:
  - sign
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print("[INFO] Fichier gtsdb.yaml créé :", yaml_path)


if __name__ == "__main__":
    convert_gtsdb_to_yolo()
    convert_ppm_to_jpg()
    create_gtsdb_yaml()