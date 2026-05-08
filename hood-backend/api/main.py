"""
main.py — Servidor FastAPI para el sistema Hood NN

Endpoint principal: POST /analyze
  → Recibe imagen JPG/PNG y retorna tabla Hood completa (10 zonas × 8 daños)

Uso:
  # Desde hood-backend/
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

Endpoints disponibles:
  GET  /              → health check
  GET  /health        → estado del servicio y modelo
  POST /analyze       → análisis automático (landmarks detectados por CV clásica)
  POST /analyze/manual → análisis con landmarks proporcionados manualmente
"""

import io
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.inference import HoodInferenceEngine, DAMAGE_TYPES
from api.zone_computer import ZONE_NAMES

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE LA APLICACIÓN
# ─────────────────────────────────────────────────────────────────────────────

ONNX_MODEL_PATH = ROOT / "models" / "hood_model.onnx"

app = FastAPI(
    title="Hood NN — Análisis de desgaste en bandeja tibial",
    description=(
        "Sistema de cuantificación automática del daño superficial en implantes "
        "de bandeja tibial de rodilla según el método Hood (10 zonas × 7 tipos de daño)."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS: permite peticiones desde el frontend (Vite en :5173 o React en :3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Motor de inferencia — inicializado en el primer uso (lazy)
_engine: Optional[HoodInferenceEngine] = None


def _get_engine() -> HoodInferenceEngine:
    """Inicialización lazy del motor de inferencia ONNX."""
    global _engine
    if _engine is None:
        if not ONNX_MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Modelo ONNX no encontrado en: {ONNX_MODEL_PATH}. "
                    "Entrena y exporta el modelo primero: "
                    "python train/train.py && python train/export_onnx.py"
                ),
            )
        _engine = HoodInferenceEngine(str(ONNX_MODEL_PATH))
    return _engine


def _decode_image(contents: bytes) -> np.ndarray:
    """Decodifica bytes de imagen a np.ndarray (H, W, 3) uint8 RGB."""
    if not _PIL_AVAILABLE:
        raise HTTPException(status_code=500, detail="Pillow no está instalado.")
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"No se pudo decodificar la imagen: {exc}",
        )


def _validate_image_type(content_type: str):
    """Valida que el Content-Type sea JPG o PNG."""
    allowed = {"image/jpeg", "image/jpg", "image/png"}
    if content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Formato no soportado: '{content_type}'. Use JPG o PNG.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/", summary="Health check básico")
def root():
    """Verifica que el servidor está activo."""
    return {
        "status":      "ok",
        "service":     "Hood NN — Tibial Tray Wear Analysis",
        "model_ready": ONNX_MODEL_PATH.exists(),
        "version":     "1.0.0",
    }


@app.get("/health", summary="Estado del servicio")
def health():
    """Retorna el estado del servicio y disponibilidad del modelo ONNX."""
    return {
        "status":       "ok",
        "model_path":   str(ONNX_MODEL_PATH),
        "model_exists": ONNX_MODEL_PATH.exists(),
        "zones":        ZONE_NAMES,
        "damage_types": DAMAGE_TYPES,
    }


@app.post(
    "/analyze",
    summary="Análisis Hood automático",
    description=(
        "Recibe una imagen JPG/PNG de un implante de bandeja tibial y retorna "
        "la tabla Hood completa con scores por zona y tipo de daño. "
        "Los landmarks se detectan automáticamente por visión computacional."
    ),
)
async def analyze_image(image: UploadFile = File(...)):
    """
    Análisis Hood con detección automática de landmarks.

    Body: multipart/form-data, campo 'image' (JPG o PNG)

    Response:
    ```json
    {
      "zones": {
        "medial_anterior":  {"delaminacion": 0, "abrasion": 1, ...},
        "medial_central":   {...},
        ...
      },
      "total_hood_score": 42,
      "landmarks": {"TL": [x, y], "TR": [x, y], ...},
      "zone_scores_matrix": [[0,1,0,...], ...]
    }
    ```
    """
    _validate_image_type(image.content_type)

    contents  = await image.read()
    image_np  = _decode_image(contents)

    try:
        engine = _get_engine()
        result = engine.analyze(image_np)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la inferencia: {exc}",
        )

    return result


@app.post(
    "/analyze/manual",
    summary="Análisis Hood con landmarks manuales",
    description=(
        "Igual que /analyze pero permite pasar los 7 landmarks anatómicos "
        "manualmente (coordenadas en píxeles). Más preciso cuando se conocen "
        "las posiciones exactas de los puntos de referencia."
    ),
)
async def analyze_with_manual_landmarks(
    image: UploadFile = File(...),
    # Esquinas del implante
    tl_x: float = Form(..., description="Top-Left X"),
    tl_y: float = Form(..., description="Top-Left Y"),
    tr_x: float = Form(..., description="Top-Right X"),
    tr_y: float = Form(..., description="Top-Right Y"),
    bl_x: float = Form(..., description="Bottom-Left X"),
    bl_y: float = Form(..., description="Bottom-Left Y"),
    br_x: float = Form(..., description="Bottom-Right X"),
    br_y: float = Form(..., description="Bottom-Right Y"),
    # Landmarks anatómicos
    mc_x: float = Form(..., description="Medial Condyle apex X"),
    mc_y: float = Form(..., description="Medial Condyle apex Y"),
    lc_x: float = Form(..., description="Lateral Condyle apex X"),
    lc_y: float = Form(..., description="Lateral Condyle apex Y"),
    ig_x: float = Form(..., description="Intercondylar Groove deepest point X"),
    ig_y: float = Form(..., description="Intercondylar Groove deepest point Y"),
):
    """
    Análisis Hood con landmarks anatómicos proporcionados por el usuario.
    Útil cuando los landmarks se han marcado con LabelMe u otra herramienta.
    """
    _validate_image_type(image.content_type)

    contents = await image.read()
    image_np = _decode_image(contents)

    landmarks = {
        "TL": [tl_x, tl_y], "TR": [tr_x, tr_y],
        "BL": [bl_x, bl_y], "BR": [br_x, br_y],
        "MC": [mc_x, mc_y], "LC": [lc_x, lc_y],
        "IG": [ig_x, ig_y],
    }

    try:
        engine = _get_engine()
        result = engine.analyze(image_np, landmarks=landmarks)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error durante la inferencia: {exc}",
        )

    return result
