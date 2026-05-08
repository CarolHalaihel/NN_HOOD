"""
dataset.py — TibialTrayDataset

Carga imágenes de implantes de bandeja tibial con sus anotaciones Hood.
Incluye augmentación agresiva para dataset pequeño (~30-40 imágenes).

Cada muestra contiene:
  - image         : tensor (3, H, W) normalizado ImageNet
  - damage_scores : tensor (10, 7) long — scores 0-3 por zona y tipo de daño
  - landmarks     : tensor (7, 2) float — coords (x, y) normalizadas [0, 1]
  - image_name    : str — nombre del archivo (para trazabilidad en LOOCV)
"""

import json
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE DOMINIO
# ─────────────────────────────────────────────────────────────────────────────

ZONE_NAMES = [
    "medial_anterior",    "medial_central",    "medial_posterior",
    "medial_periferico",  "surco_medial",
    "lateral_anterior",   "lateral_central",   "lateral_posterior",
    "lateral_periferico", "surco_lateral",
]

DAMAGE_TYPES = [
    "delaminacion", "abrasion",    "rayado",      "brunido",
    "picado",       "residuos",    "deformacion",
]

LANDMARK_NAMES = ["TL", "TR", "BL", "BR", "MC", "LC", "IG"]


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMACIONES
# ─────────────────────────────────────────────────────────────────────────────

def build_train_transform(image_size: int = 224) -> A.Compose:
    """
    Augmentación agresiva para dataset pequeño.
    Multiplica efectivamente las muestras ×30-50 por época.
    Incluye keypoints para mantener coherencia landmarks ↔ imagen.

    Transformaciones geométricas: flip, rotación ±15°, perspectiva ligera.
    Transformaciones fotométricas: brillo, contraste, CLAHE, ruido, blur.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),

            # ── Geométricas ──────────────────────────────────────────────────
            # Flip horizontal moderado (implantes son casi simétricos bilaterales)
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=15, border_mode=cv2.BORDER_REFLECT_101, p=0.7),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=0,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.5,
            ),
            A.Perspective(scale=(0.02, 0.05), p=0.3),

            # ── Fotométricas ─────────────────────────────────────────────────
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
            A.HueSaturationValue(
                hue_shift_limit=5, sat_shift_limit=20, val_shift_limit=20, p=0.5
            ),
            # CLAHE mejora contraste local — útil para detectar desgaste superficial
            A.CLAHE(clip_limit=3.0, p=0.3),
            A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.ImageCompression(quality_lower=85, quality_upper=100, p=0.2),

            # ── Normalización ImageNet ────────────────────────────────────────
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def build_val_transform(image_size: int = 224) -> A.Compose:
    """Transformación de validación/inferencia: solo resize + normalización."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def build_landmark_transform(image_size: int = 512) -> A.Compose:
    """
    Transformación para el LandmarkDetector (entrada 512×512).
    Augmentación más conservadora para preservar precisión de localización.
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DATASET PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class TibialTrayDataset(Dataset):
    """
    Dataset de implantes de bandeja tibial con anotaciones Hood.

    Parámetros:
      images_dir       : directorio con las imágenes JPG/PNG
      annotations_path : ruta al archivo annotations.json
      transform        : transformación Albumentations (opcional, por defecto train o val)
      indices          : subset de índices para LOOCV (None = todos)
      image_size       : tamaño de salida cuadrado
      augment          : si True y transform es None, usa build_train_transform
    """

    def __init__(
        self,
        images_dir: str,
        annotations_path: str,
        transform: Optional[A.Compose] = None,
        indices: Optional[list] = None,
        image_size: int = 224,
        augment: bool = True,
    ):
        self.images_dir = Path(images_dir)
        self.image_size = image_size

        # Cargar anotaciones desde JSON
        with open(annotations_path, "r", encoding="utf-8") as f:
            all_annotations = json.load(f)

        # Filtrar solo imágenes que existen en disco
        self.samples = []
        for img_name, ann in all_annotations.items():
            img_path = self.images_dir / img_name
            if img_path.exists():
                self.samples.append((img_name, ann))
            else:
                print(f"[AVISO] Imagen no encontrada en disco, se omite: {img_path}")

        if not self.samples:
            raise RuntimeError(
                f"No se encontró ninguna imagen en {self.images_dir}. "
                "Verifica que annotations.json y las imágenes estén en las rutas correctas."
            )

        # Subset para LOOCV: permite dejar fuera una imagen como validación
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

        # Transformación
        if transform is not None:
            self.transform = transform
        elif augment:
            self.transform = build_train_transform(image_size)
        else:
            self.transform = build_val_transform(image_size)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_name, ann = self.samples[idx]
        img_path = self.images_dir / img_name

        # ── 1. Cargar imagen ──────────────────────────────────────────────────
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f"No se puede leer la imagen: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = image.shape[:2]

        # ── 2. Landmarks como lista de (x, y) para Albumentations ────────────
        raw_landmarks = ann.get("landmarks", {})
        keypoints = []
        for name in LANDMARK_NAMES:
            if name in raw_landmarks:
                x, y = raw_landmarks[name]
                # Clamp por seguridad a los bordes de la imagen
                x = float(max(0, min(w_orig - 1, x)))
                y = float(max(0, min(h_orig - 1, y)))
                keypoints.append((x, y))
            else:
                # Landmark ausente: usar centro como fallback
                keypoints.append((float(w_orig / 2), float(h_orig / 2)))

        # ── 3. Aplicar transformaciones ───────────────────────────────────────
        transformed      = self.transform(image=image, keypoints=keypoints)
        image_tensor     = transformed["image"]      # (3, H, W)
        kps_transformed  = transformed["keypoints"]  # lista de (x, y)

        # Normalizar landmarks al rango [0, 1] respecto al tamaño de salida
        landmarks = torch.zeros(len(LANDMARK_NAMES), 2, dtype=torch.float32)
        for i, (x, y) in enumerate(kps_transformed):
            landmarks[i, 0] = float(x) / self.image_size
            landmarks[i, 1] = float(y) / self.image_size
        landmarks = landmarks.clamp(0.0, 1.0)

        # ── 4. Damage scores (10 zonas × 7 daños) ────────────────────────────
        raw_scores    = ann.get("damage_scores", {})
        damage_scores = torch.zeros(10, 7, dtype=torch.long)
        for zone_idx in range(10):
            key = f"zona_{zone_idx}"
            if key in raw_scores:
                for dmg_idx, score in enumerate(raw_scores[key][:7]):
                    # Clamp a rango válido [0, 3]
                    damage_scores[zone_idx, dmg_idx] = int(max(0, min(3, int(score))))

        return {
            "image":         image_tensor,    # (3, H, W) float32
            "damage_scores": damage_scores,   # (10, 7)   long
            "landmarks":     landmarks,       # (7, 2)    float32 en [0, 1]
            "image_name":    img_name,        # str
        }

    def get_class_weights(self) -> torch.Tensor:
        """
        Calcula pesos de clase para CrossEntropyLoss basado en distribución
        de scores en el dataset. Útil para mitigar el desbalance (score 0 dominante).

        Retorna: tensor (4,) con pesos para scores 0, 1, 2, 3
        """
        counts = torch.zeros(4, dtype=torch.float32)
        for _, ann in self.samples:
            raw_scores = ann.get("damage_scores", {})
            for zone_key in raw_scores.values():
                for score in zone_key:
                    score_int = int(max(0, min(3, int(score))))
                    counts[score_int] += 1

        # Peso inverso a frecuencia (normalizado)
        counts = counts.clamp(min=1)
        weights = 1.0 / counts
        weights = weights / weights.sum() * 4  # normalizar a media=1
        return weights
