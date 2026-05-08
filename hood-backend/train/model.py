"""
model.py — Definición de los dos modelos del sistema Hood NN

  - LandmarkDetector : EfficientNet-B2 (features_only) + FPN → 7 heatmaps gaussianos
  - HoodNet          : DINOv2 vits14 (congelado) + 8 cabezas lineales → scores Hood (0-3)

Decisión de arquitectura:
  - DINOv2 en HoodNet: solo 12.288 parámetros entrenables → ideal para dataset <40 imágenes en CPU.
    El backbone extrae representaciones ricas sin haber visto implantes; solo las cabezas aprenden.
  - EfficientNet-B2 en LandmarkDetector: la arquitectura CNN con FPN es natural para
    localización espacial precisa (heatmaps gaussianos).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ─────────────────────────────────────────────────────────────────────────────
# UTILIDADES COMPARTIDAS
# ─────────────────────────────────────────────────────────────────────────────

def soft_argmax2d(heatmaps: torch.Tensor) -> torch.Tensor:
    """
    Extrae coordenadas (x, y) normalizadas [0, 1] de heatmaps mediante soft-argmax.
    Diferenciable → permite backpropagation a través de las coordenadas.

    Entrada:  (B, N, H, W)  — N heatmaps de tamaño HxW
    Salida:   (B, N, 2)     — coordenadas (x, y) en [0, 1]
    """
    B, N, H, W = heatmaps.shape

    # Aplanar y normalizar con softmax para obtener distribución de probabilidad
    flat    = heatmaps.reshape(B, N, -1)
    weights = F.softmax(flat, dim=-1).reshape(B, N, H, W)

    # Grids de coordenadas normalizadas [0, 1]
    ys = torch.linspace(0, 1, H, device=heatmaps.device)
    xs = torch.linspace(0, 1, W, device=heatmaps.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W) cada uno

    # Expectativa (coordenada esperada)
    coord_x = (weights * grid_x.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))  # (B, N)
    coord_y = (weights * grid_y.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))  # (B, N)

    return torch.stack([coord_x, coord_y], dim=-1)  # (B, N, 2)


# ─────────────────────────────────────────────────────────────────────────────
# MODELO A — DETECTOR DE LANDMARKS
# ─────────────────────────────────────────────────────────────────────────────

class FPNDecoder(nn.Module):
    """
    Feature Pyramid Network simplificado.
    Recibe 4 mapas de características del backbone y los fusiona de arriba abajo
    con upsampling bilineal para preservar detalle espacial fino.
    """

    def __init__(self, in_channels: list, out_channels: int = 128):
        super().__init__()
        # Proyección lateral 1×1 para reducir canales de cada escala
        self.lateral = nn.ModuleList([
            nn.Conv2d(c, out_channels, kernel_size=1) for c in in_channels
        ])
        # Convolución de refinamiento 3×3 post-fusión
        self.refine = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_channels
        ])

    def forward(self, features: list) -> torch.Tensor:
        """
        features: lista de tensores (B, C_i, H_i, W_i), de resolución
                  menor a mayor (índice 0 = más baja resolución).
        """
        # Proyección lateral en todas las escalas
        laterals = [lat(f) for lat, f in zip(self.lateral, features)]

        # Fusión top-down: la escala de menor resolución guía a la siguiente
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        # Retornar el mapa de mayor resolución (índice 0) refinado
        return self.refine[0](laterals[0])


class LandmarkDetector(nn.Module):
    """
    Detector de 7 landmarks anatómicos en la imagen completa de la bandeja tibial.

    Entrada:  (B, 3, 512, 512)
    Salidas:
      - heatmaps : (B, 7, 64, 64)   — mapas de calor gaussianos (uno por landmark)
      - coords   : (B, 7, 2)         — coordenadas (x, y) normalizadas [0, 1]

    Los 7 landmarks:
      TL, TR, BL, BR → 4 esquinas del implante
      MC             → cima del cóndilo medial
      LC             → cima del cóndilo lateral
      IG             → punto más profundo del surco intercondíleo
    """

    NUM_LANDMARKS  = 7
    LANDMARK_NAMES = ["TL", "TR", "BL", "BR", "MC", "LC", "IG"]

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # Backbone EfficientNet-B2 en modo features_only (4 escalas de resolución)
        # Escalas de salida: /4, /8, /16, /32 respecto a la imagen de entrada
        self.backbone = timm.create_model(
            "efficientnet_b2",
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
        )

        # Canales de salida del backbone B2 por escala: [24, 48, 120, 352]
        backbone_channels = self.backbone.feature_info.channels()

        # FPN decoder: fusiona las 4 escalas
        self.fpn = FPNDecoder(in_channels=backbone_channels, out_channels=128)

        # Cabeza final: proyecta 128 canales a 7 heatmaps
        self.head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, self.NUM_LANDMARKS, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        """
        x: (B, 3, 512, 512)

        Retorna:
          heatmaps: (B, 7, 64, 64) — logits de heatmap (sin activación)
          coords:   (B, 7, 2)      — coordenadas normalizadas [0, 1]
        """
        # Extracción multi-escala
        features = self.backbone(x)  # lista de 4 tensores

        # FPN → mapa de 128 canales a ~resolución /4 (128×128 para entrada 512)
        fpn_out = self.fpn(features)

        # Reducir a 64×64 para compatibilidad con targets gaussianos
        fpn_out = F.interpolate(
            fpn_out, size=(64, 64), mode="bilinear", align_corners=False
        )

        # Heatmaps (logits — la loss usa activación internamente si es necesario)
        heatmaps = self.head(fpn_out)  # (B, 7, 64, 64)

        # Coordenadas diferenciables vía soft-argmax
        coords = soft_argmax2d(heatmaps)  # (B, 7, 2)

        return heatmaps, coords


# ─────────────────────────────────────────────────────────────────────────────
# MODELO B — HOODNET (clasificador de daño por zona)
# ─────────────────────────────────────────────────────────────────────────────

class HoodNet(nn.Module):
    """
    Clasificador de daño Hood por zona recortada de la bandeja tibial.

    Backbone: DINOv2 ViT-S/14 (Meta, 2023) completamente congelado.
    Solo se entrenan 7 cabezas lineales: 7 × Linear(384→4) = 10.752 parámetros.
    Esto permite entrenamiento en CPU en ~5 minutos con dataset de 30-40 imágenes.

    Entrada:  (B, 3, 224, 224)  — recorte de zona, normalizado ImageNet
    Salida:   (B, 7, 4)          — logits para score 0/1/2/3 × 7 tipos de daño

    Los 7 tipos de daño (en orden):
      0: delaminacion  1: abrasion  2: rayado   3: brunido
      4: picado        5: residuos  6: deformacion
    """

    DAMAGE_TYPES = [
        "delaminacion", "abrasion", "rayado", "brunido",
        "picado", "residuos", "deformacion",
    ]
    NUM_DAMAGE  = 7
    NUM_SCORES  = 4   # clases: 0, 1, 2, 3
    FEATURE_DIM = 384  # DINOv2 ViT-S/14

    def __init__(self):
        super().__init__()

        # Cargar DINOv2 ViT-S/14 desde torch.hub
        # Se descarga automáticamente la primera vez (~85 MB)
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            "dinov2_vits14",
            pretrained=True,
            verbose=False,
        )

        # Congelar TODOS los parámetros del backbone
        # → solo las cabezas lineales reciben gradientes
        for param in self.backbone.parameters():
            param.requires_grad = False

        # 8 cabezas independientes: Linear(384→4), una por tipo de daño
        # Independientes → cada daño tiene su propio clasificador
        self.heads = nn.ModuleList([
            nn.Linear(self.FEATURE_DIM, self.NUM_SCORES)
            for _ in range(self.NUM_DAMAGE)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 224, 224)
        Retorna: (B, 8, 4) — logits por tipo de daño × clase de score
        """
        # Extracción de features con DINOv2 (sin gradientes en backbone)
        with torch.no_grad():
            features = self.backbone(x)  # (B, 384)

        # Clasificación independiente por cada tipo de daño
        logits = torch.stack(
            [head(features) for head in self.heads], dim=1
        )  # (B, 8, 4)

        return logits

    def predict_scores(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predicción directa de scores (0-3) sin gradientes.
        Útil para evaluación rápida.

        Retorna: (B, 8) con scores predichos 0, 1, 2 o 3
        """
        with torch.no_grad():
            logits = self.forward(x)      # (B, 8, 4)
            scores = logits.argmax(dim=-1)  # (B, 8)
        return scores


# ─────────────────────────────────────────────────────────────────────────────
# WING LOSS — pérdida para entrenamiento de landmarks
# ─────────────────────────────────────────────────────────────────────────────

class WingLoss(nn.Module):
    """
    Wing Loss para regresión de landmarks (Feng et al., 2018).
    Más sensible que MSE para errores pequeños (rango no-lineal cerca de 0).

    Fórmula:
      WL(x) = w * ln(1 + |x|/ε)          si |x| < w
              |x| - C                     en caso contrario
    donde C = w - w*ln(1 + w/ε) es la constante de continuidad.

    Referencia: "Wing Loss for Robust Facial Landmark Localisation"
    """

    def __init__(self, w: float = 10.0, epsilon: float = 2.0):
        super().__init__()
        self.w       = w
        self.epsilon = epsilon
        # Constante de continuidad en el punto de transición
        self.C = w - w * (1.0 + w / epsilon) ** (-1) * (w / epsilon)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pred, target: (B, N, 2) — coordenadas normalizadas [0, 1]
        Retorna: escalar (pérdida media sobre todos los landmarks y coordenadas)
        """
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1.0 + diff / self.epsilon),
            diff - self.C,
        )
        return loss.mean()
