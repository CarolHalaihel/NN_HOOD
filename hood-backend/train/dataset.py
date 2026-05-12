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
import site as _site
import sys
from pathlib import Path
from typing import Optional

# Asegurar que user site-packages está en sys.path (albumentations, torch, etc.)
_usp = _site.getusersitepackages()
if _usp not in sys.path:
    sys.path.insert(0, _usp)

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

# HoodZoneComputer vive en api/ — añadir la raíz del proyecto al path
_BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))
from api.zone_computer import HoodZoneComputer


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
    "rayado",       "picado",      "brunido",     "abrasion",
    "delaminacion", "deformacion", "residuos",
]

LANDMARK_NAMES = ["TL", "TR", "BL", "BR", "MC", "LC", "IG"]


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMACIONES
# ─────────────────────────────────────────────────────────────────────────────

def build_train_transform(image_size: int = 224) -> A.Compose:
    """
    Augmentación intensiva para dataset pequeño de implantes tibiales.

    Geométricas:
      - Flip H (implantes casi simétricos), Flip V raro
      - Rotate ±30°, ShiftScaleRotate (traslación ±8%, escala ±15%)
      - Perspective, GridDistortion, OpticalDistortion (variación de cámara/superficie)
      - CoarseDropout (simula tornillos y oclusiones parciales de zona)

    Fotométricas (superficie metálica con reflejos y colores variables):
      - RandomBrightnessContrast agresivo, RandomGamma
      - HueSaturationValue (temperatura de color)
      - CLAHE (resaltar micro-texturas de desgaste)
      - ToGray ocasional (invariancia al color)
      - GaussNoise, SaltAndPepper
      - OneOf[GaussianBlur | Sharpen | MotionBlur]
      - ImageCompression (artefactos JPEG)
      - RandomShadow (iluminación parcial)
    """
    return A.Compose(
        [
            A.Resize(image_size, image_size),

            # ── Geométricas ──────────────────────────────────────────────────
            A.HorizontalFlip(p=0.4),
            A.VerticalFlip(p=0.1),
            A.Rotate(limit=30, p=0.8),
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.15,
                rotate_limit=0,
                p=0.6,
            ),
            A.Perspective(scale=(0.02, 0.06), p=0.4),
            A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.3),
            A.OpticalDistortion(distort_limit=0.1, p=0.2),
            # Simula tornillos y oclusiones parciales de la zona evaluada
            A.CoarseDropout(
                num_holes_range=(1, 6),
                hole_height_range=(0.05, 0.10),
                hole_width_range=(0.05, 0.10),
                fill=0, p=0.3,
            ),

            # ── Fotométricas ─────────────────────────────────────────────────
            A.RandomBrightnessContrast(brightness_limit=0.35, contrast_limit=0.35, p=0.8),
            A.RandomGamma(gamma_limit=(70, 140), p=0.4),
            A.HueSaturationValue(
                hue_shift_limit=8, sat_shift_limit=30, val_shift_limit=30, p=0.6
            ),
            # CLAHE: resalta micro-texturas de desgaste superficial
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
            # Ocasionalmente entrenar en gris → invariancia al color del implante
            A.ToGray(p=0.1),
            A.GaussNoise(std_range=(0.04, 0.20), p=0.4),
            A.SaltAndPepper(amount=(0.001, 0.01), p=0.2),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.2), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.4),
            A.ImageCompression(quality_range=(75, 100), p=0.3),
            A.RandomShadow(
                shadow_roi=(0, 0, 1, 1),
                num_shadows_limit=(1, 2),
                shadow_dimension=4, p=0.2,
            ),

            # ── Normalización ImageNet ────────────────────────────────────────
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
    )


def build_val_transform(image_size: int = 224) -> A.Compose:
    """Transformación de validación/inferencia: solo resize + normalización."""
    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ],
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
            A.GaussNoise(std_range=(0.02, 0.08), p=0.2),
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
        only_aE: bool = False,
        repeat_factor: int = 1,
    ):
        self.images_dir = Path(images_dir)
        self.image_size = image_size

        # Cargar anotaciones desde JSON
        with open(annotations_path, "r", encoding="utf-8") as f:
            all_annotations = json.load(f)

        # Filtrar imágenes que existen en disco (y opcionalmente solo _aE)
        self.samples = []
        for img_name, ann in all_annotations.items():
            if img_name.startswith("_"):
                continue
            # Filtro opcional: solo vista superior (_aE.png) con 10 zonas Hood
            if only_aE and not img_name.endswith("_aE.png"):
                continue
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

        # Upsampling virtual: repite cada muestra repeat_factor veces por época.
        # Cada repetición recibe una augmentación aleatoria diferente → más variedad
        # sin duplicar imágenes en disco. Con 13 imágenes × 8 = 104 muestras/época.
        if repeat_factor > 1:
            self.samples = self.samples * repeat_factor

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


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE CARGA
# ─────────────────────────────────────────────────────────────────────────────

def load_base_samples(
    images_dir: Path,
    annotations_path: Path,
    only_aE: bool = True,
) -> list:
    """
    Carga la lista base de (img_name, ann) que tienen:
      - imagen en disco
      - los 7 landmarks requeridos por HoodZoneComputer
    Devuelve una lista de tuplas [(img_name, ann), ...] con imágenes únicas.
    """
    with open(annotations_path, "r", encoding="utf-8") as f:
        all_ann = json.load(f)

    samples = []
    for img_name, ann in all_ann.items():
        if img_name.startswith("_"):
            continue
        if only_aE and not img_name.endswith("_aE.png"):
            continue
        img_path = images_dir / img_name
        if not img_path.exists():
            print(f"[AVISO] Imagen no encontrada: {img_path}")
            continue
        lm = ann.get("landmarks", {})
        missing = [k for k in ("TL", "TR", "BL", "BR", "MC", "LC", "IG") if k not in lm]
        if missing:
            print(f"[AVISO] Landmarks {missing} ausentes en {img_name}, se omite")
            continue
        samples.append((img_name, ann))

    return samples


# ─────────────────────────────────────────────────────────────────────────────
# DATASET POR ZONA (RECORTES)
# ─────────────────────────────────────────────────────────────────────────────

class ZoneCropDataset(Dataset):
    """
    Dataset que expone un recorte de zona como muestra de entrenamiento.

    Cada muestra corresponde a un par (imagen, zona_idx):
      - La imagen se carga desde disco.
      - HoodZoneComputer extrae el recorte de la zona usando los landmarks.
      - El recorte pasa por la transformación de augmentación.
      - Los scores de daño son los 7 valores de esa zona concreta.

    Parámetros:
      images_dir    : directorio con imágenes PNG/JPG
      base_samples  : lista de (img_name, ann) ya filtrada (de load_base_samples)
      zone_idx      : índice de zona Hood 0–9
      transform     : transformación Albumentations personalizada (o None)
      image_size    : tamaño de salida cuadrado
      augment       : si True y transform es None, usa build_train_transform
      repeat_factor : multiplica las muestras para augmentación virtual por época
    """

    def __init__(
        self,
        images_dir: str,
        base_samples: list,
        zone_idx: int,
        transform: Optional[A.Compose] = None,
        image_size: int = 224,
        augment: bool = True,
        repeat_factor: int = 1,
    ):
        self.images_dir = Path(images_dir)
        self.image_size = image_size
        self.zone_idx   = zone_idx

        if not (0 <= zone_idx <= 9):
            raise ValueError(f"zone_idx debe estar entre 0 y 9, recibido: {zone_idx}")

        self._base   = list(base_samples)
        self.samples = self._base * repeat_factor

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

        # ── 2. Extraer recorte de zona mediante HoodZoneComputer ──────────────
        landmarks = ann.get("landmarks", {})
        try:
            zc   = HoodZoneComputer(landmarks)
            crop = zc.get_zone_crop(image, self.zone_idx, output_size=None)
            if crop is None or crop.size == 0 or min(crop.shape[:2]) == 0:
                raise ValueError("recorte vacío")
        except Exception as e:
            print(f"[AVISO] Crop zona {self.zone_idx} de {img_name} falló ({e}), "
                  "usando imagen completa como fallback")
            crop = image

        # ── 3. Augmentación / transformación ─────────────────────────────────
        transformed  = self.transform(image=crop)
        image_tensor = transformed["image"]   # (3, H, W) float32

        # ── 4. Scores de daño para esta zona (7 valores) ─────────────────────
        zone_key  = f"zona_{self.zone_idx}"
        raw       = ann.get("damage_scores", {}).get(zone_key, [0] * 7)
        scores    = torch.zeros(7, dtype=torch.long)
        for i, s in enumerate(raw[:7]):
            scores[i] = int(max(0, min(3, int(s))))

        return {
            "image":         image_tensor,    # (3, H, W) float32
            "damage_scores": scores,          # (7,)      long
            "zone_idx":      self.zone_idx,   # int
            "image_name":    img_name,        # str
        }
