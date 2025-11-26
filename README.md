# The Best Autonomous Car – Perception (Traffic Signs)

Ce dépôt contient la partie **Perception** de notre projet de voiture autonome, centrée sur :

- la **classification de panneaux de signalisation** avec un CNN (Keras / TensorFlow),
- la **détection de panneaux** avec YOLOv8 (Ultralytics),
- un **pipeline fusion YOLO + CNN** : détection → recadrage → classification → gestion des panneaux inconnus.

---

## 1. Vue d’ensemble

La perception des panneaux est découpée en trois briques principales :

1. **CNN – Classification (Keras / TensorFlow)**  
   - Entrée : image de panneau recadrée (32×32, RGB).  
   - Sortie : probabilité sur 5 classes :

     | Classe GTSRB | Signification            |
     |-------------|--------------------------|
     | `00000`     | Speed limit 20 km/h      |
     | `00002`     | Speed limit 50 km/h      |
     | `00014`     | Stop                     |
     | `00033`     | Turn right               |
     | `00034`     | Turn left                |

2. **YOLOv8 – Détection (Ultralytics)**  
   - Entrée : image complète ou frame vidéo (scène de route).  
   - Sortie : bounding boxes de panneaux (`class = sign`) avec score de confiance.

3. **Pipeline YOLO + CNN – Fusion**  
   - YOLO détecte les panneaux et donne des boîtes englobantes.  
   - Pour chaque boîte :
     - on découpe le patch dans l’image originale,
     - on le passe dans le CNN,
     - si le panneau ne fait pas partie des 5 classes apprises ou si la confiance est faible → on renvoie **"Inconnu"**.

---

## 2. Organisation du projet (perception)

Arborescence (simplifiée) :

```text
The-best-autonomous-car/
├── models/
│   ├── My_GTSRB_5classes_balanced.h5   # CNN entraîné sur 5 classes GTSRB
│   └── best.pt                         # YOLOv8 entraîné sur GTSDB (détection de panneaux)
│
├── src/
│   └── perception/
│       ├── pipeline_detection.py       # Pipeline YOLO + CNN (détection + classification)
│       ├── gtsdb_to_yolo.py           # (optionnel) conversion GTSDB -> format YOLO
│       ├── train_cnn_gtsrb.py         # (optionnel) entraînement CNN sur GTSRB
│       └── test_cnn_image.py          # (optionnel) test du CNN sur une image
│
├── assets/
│   └── perception/
│       ├── example_sign_detection.jpg  # image d'exemple avec panneau(x)
│       ├── example_sign_detection.mp4  # vidéo d'exemple
│       ├── output_cnn_yolo_image.jpg   # image annotée générée par le pipeline
│       └── output_cnn_yolo_video.mp4   # vidéo annotée générée par le pipeline
│
└── README.md


	
