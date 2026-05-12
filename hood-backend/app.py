"""
app.py  Interfaz web para Hood NN

Aplicación Streamlit completa para gestionar el ciclo de vida del sistema:
  📁 Datos         subir imágenes, editar anotaciones Hood (tabla interactiva)
  ✅️ Entrenamiento  configurar, lanzar y monitorear el entrenamiento en tiempo real
   Resultados    ver métricas LOOCV con gráficos interactivos
  ✅ Inferencia    probar el modelo con imágenes nuevas

Inicio:
  cd hood-backend
  streamlit run app.py
"""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    _PLOTLY = True
except ImportError:
    _PLOTLY = False

try:
    import cv2 as cv2_
    from PIL import Image as PILImage
    _CV2_AVAILABLE = True
except ImportError:
    _CV2_AVAILABLE = False

try:
    from streamlit_image_coordinates import streamlit_image_coordinates as sic
    _SIC_AVAILABLE = True
except ImportError:
    _SIC_AVAILABLE = False

# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
# RUTAS Y CONSTANTES
# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅

ROOT             = Path(__file__).resolve().parent
IMAGES_DIR       = ROOT / "data" / "images"
ANNOTATIONS_FILE = ROOT / "data" / "annotations.json"
CROPS_DIR        = ROOT / "data" / "crops"
AUG_CROPS_DIR    = ROOT / "data" / "augmented_crops"
MODELS_DIR       = ROOT / "models"
PROGRESS_FILE    = MODELS_DIR / "training_progress.json"
LOG_FILE         = MODELS_DIR / "training.log"
PID_FILE         = MODELS_DIR / "training.pid"
CHECKPOINT_FILE  = MODELS_DIR / "training_checkpoint.pt"
STOP_FILE        = MODELS_DIR / "training_stop.flag"
LOOCV_FILE       = MODELS_DIR / "loocv_results.json"
ONNX_MODEL       = MODELS_DIR / "hood_model.onnx"
PT_MODEL         = MODELS_DIR / "hoodnet_final.pt"

IMAGES_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)
AUG_CROPS_DIR.mkdir(parents=True, exist_ok=True)

ZONE_NAMES = [
    "medial_periferico",  "medial_anterior",   "medial_central",    "medial_posterior",
    "lateral_central",    "lateral_anterior",  "lateral_periferico","lateral_posterior",
    "surco_anterior",     "surco_posterior",
]

# Posiciones por defecto de cada zona como fracción (fx, fy) del tamaño de imagen.
# Calibradas para rodilla DERECHA vista desde arriba (medial a la derecha).
# Para rodilla izquierda se espeja fx  (1 - fx).
DEFAULT_ZONE_FRACS_RIGHT = [
    [0.663, 0.560],  # 0 · medial_periferico   (center-right, mid)
    [0.720, 0.240],  # 1 · medial_anterior      (upper-right)
    [0.806, 0.560],  # 2 · medial_central       (rightmost, mid)
    [0.720, 0.846],  # 3 · medial_posterior     (lower-right)
    [0.120, 0.560],  # 4 · lateral_central      (leftmost, mid)
    [0.191, 0.257],  # 5 · lateral_anterior     (upper-left)
    [0.277, 0.560],  # 6 · lateral_periferico   (center-left, mid)
    [0.191, 0.846],  # 7 · lateral_posterior    (lower-left)
    [0.443, 0.296],  # 8 · surco_anterior       (center, upper)
    [0.443, 0.714],  # 9 · surco_posterior      (center, lower)
]

DAMAGE_TYPES = [
    "rayado",       "picado",      "brunido",     "abrasion",
    "delaminacion", "deformacion", "residuos",
]

LANDMARK_NAMES = ["TL", "TR", "BL", "BR", "MC", "LC", "IG"]

# Etiquetas de las 10 zonas Hood para el interfaz de marcado
ZONE_CLICK_LABELS = {
    "zona_0": "0 · Medial periférico",
    "zona_1": "1 · Medial anterior",
    "zona_2": "2 · Medial central",
    "zona_3": "3 · Medial posterior",
    "zona_4": "4 · Lateral central",
    "zona_5": "5 · Lateral anterior",
    "zona_6": "6 · Lateral periférico",
    "zona_7": "7 · Lateral posterior",
    "zona_8": "8 · Surco anterior",
    "zona_9": "9 · Surco posterior",
}

# Colores BGR para los círculos de zona en la imagen de clic
ZONE_CV_COLORS = [
    (200, 220, 255), (100, 180, 255), (50, 130, 255), (0,  80, 200),   # 0-3 medial
    (255, 130,  80), (255, 180, 100), (220, 80,  50), (180, 50,  30),  # 4-7 lateral
    (255, 240, 100), (255, 200,  50),                                    # 8-9 surcos
]

SCORE_COLORS = {
    0: "#f5f5f5",   # gris claro
    1: "#fff176",   # amarillo
    2: "#ffb74d",   # naranja
    3: "#ef5350",   # rojo
}


# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
# HELPERS
# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅

def load_annotations() -> dict:
    """Carga annotations.json filtrando claves internas (prefijo _)."""
    if ANNOTATIONS_FILE.exists():
        try:
            with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            return {k: v for k, v in data.items() if not k.startswith("_")}
        except Exception:
            return {}
    return {}


def save_annotations(annotations: dict):
    """Guarda annotations.json manteniendo la clave _INSTRUCCIONES si existe."""
    # Preservar instrucciones originales
    original = {}
    if ANNOTATIONS_FILE.exists():
        try:
            with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
                original = json.load(f)
        except Exception:
            pass
    instructions = {k: v for k, v in original.items() if k.startswith("_")}

    combined = {**annotations, **instructions}
    with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)


def get_image_list() -> list:
    """Lista todas las imágenes JPG/PNG en data/images/."""
    exts = {".jpg", ".jpeg", ".png"}
    return sorted([
        f.name for f in IMAGES_DIR.iterdir()
        if f.suffix.lower() in exts
    ])


def is_training_running() -> bool:
    """Verifica si el proceso de entrenamiento sigue activo."""
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, 0)  # señal 0 = solo verificar existencia del proceso
        return True
    except (ValueError, OSError, PermissionError):
        return False


def has_checkpoint() -> bool:
    """Verifica si existe un checkpoint de entrenamiento pausado."""
    return CHECKPOINT_FILE.exists()


def read_progress() -> dict:
    """Lee el archivo de progreso JSON escrito por train.py."""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _fmt_eta(seconds: float) -> str:
    """Convierte segundos a texto legible: 1h 23m 45s."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s   = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def read_log_tail(n_lines: int = 25) -> str:
    """Lee las últimas N líneas del log de entrenamiento."""
    if LOG_FILE.exists():
        try:
            lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(lines[-n_lines:])
        except Exception:
            pass
    return "(sin log aún)"


def _get_image_type(img_name: str) -> str:
    """Detecta si la imagen es de arriba (_aE) o abajo (_bE)."""
    if img_name.endswith("_aE.png"):
        return "arriba"
    elif img_name.endswith("_bE.png"):
        return "abajo"
    else:
        return "arriba"  # Por defecto


def _default_annotation(img_name: str) -> dict:
    """Genera una anotación vacía por defecto según el tipo de imagen."""
    img_type = _get_image_type(img_name)
    
    if img_type == "abajo":
        # Para vista inferior: 4 puntos (AM, AL, PM, PL)
        return {
            "knee_side":     "derecha",
            "image_type":    "abajo",
            "landmarks":     {lm: [0.0, 0.0] for lm in ["AM", "AL", "PM", "PL"]},
            "damage_scores": {f"zona_{z}": [0] * 7 for z in range(4)},
        }
    else:
        # Para vista superior: 7 landmarks Hood estándar
        return {
            "knee_side":     "derecha",
            "image_type":    "arriba",
            "landmarks":     {lm: [0.0, 0.0] for lm in LANDMARK_NAMES},
            "damage_scores": {f"zona_{z}": [0] * 7 for z in range(10)},
        }


def zone_centers_to_landmarks(zone_centers: dict, img_w: int, img_h: int) -> dict:
    """
    Deriva los 7 landmarks anatómicos a partir de los 10 centros de zona Hood.

    Lógica:
      MC  = centroide de las zonas mediales internas (1, 2, 3)
      LC  = centroide de las zonas laterales internas (4, 5, 7)
      IG  = punto medio entre surco medial (8) y surco lateral (9)
      TL/BL = esquinas izquierdas estimadas por extensión del lado medial
      TR/BR = esquinas derechas estimadas por extensión del lado lateral
    """
    def pt(key):
        c = zone_centers.get(key, [img_w / 2.0, img_h / 2.0])
        return np.array(c, dtype=float)

    MC = (pt("zona_1") + pt("zona_2") + pt("zona_3")) / 3.0
    LC = (pt("zona_4") + pt("zona_5") + pt("zona_7")) / 3.0
    IG = (pt("zona_8") + pt("zona_9")) / 2.0

    med_pts = [pt(f"zona_{i}") for i in [0, 1, 2, 3, 8]]
    lat_pts = [pt(f"zona_{i}") for i in [4, 5, 6, 7, 9]]
    all_pts = med_pts + lat_pts

    all_xs = [p[0] for p in all_pts]
    all_ys = [p[1] for p in all_pts]

    med_x_mean = float(np.mean([p[0] for p in med_pts]))
    lat_x_mean = float(np.mean([p[0] for p in lat_pts]))
    left_pts   = med_pts if med_x_mean <= lat_x_mean else lat_pts
    right_pts  = lat_pts if med_x_mean <= lat_x_mean else med_pts

    margin_x = max(10.0, (max(all_xs) - min(all_xs)) * 0.20)
    margin_y = max(10.0, (max(all_ys) - min(all_ys)) * 0.25)

    TL = [max(0.0, min(p[0] for p in left_pts)  - margin_x), max(0.0, min(all_ys) - margin_y)]
    BL = [max(0.0, min(p[0] for p in left_pts)  - margin_x), min(float(img_h - 1), max(all_ys) + margin_y)]
    TR = [min(float(img_w - 1), max(p[0] for p in right_pts) + margin_x), max(0.0, min(all_ys) - margin_y)]
    BR = [min(float(img_w - 1), max(p[0] for p in right_pts) + margin_x), min(float(img_h - 1), max(all_ys) + margin_y)]

    return {
        "TL": TL, "TR": TR, "BL": BL, "BR": BR,
        "MC": [float(MC[0]), float(MC[1])],
        "LC": [float(LC[0]), float(LC[1])],
        "IG": [float(IG[0]), float(IG[1])],
    }


def landmarks_to_zone_centers(landmarks: dict) -> dict:
    """
    Obtiene los 10 centros de zona Hood a partir de los landmarks anatómicos guardados.
    Usa HoodZoneComputer para calcular los bboxes y toma sus centros.
    """
    try:
        sys.path.insert(0, str(ROOT))
        from api.zone_computer import HoodZoneComputer
        zc = HoodZoneComputer(landmarks)
        centers = {}
        for i in range(10):
            x1, y1, x2, y2 = zc.get_zone_bbox(i)
            centers[f"zona_{i}"] = [float((x1 + x2) / 2), float((y1 + y2) / 2)]
        return centers
    except Exception:
        return {f"zona_{i}": [0.0, 0.0] for i in range(10)}


def four_landmarks_to_zone_centers(landmarks: dict) -> dict:
    """
    Obtiene los 4 centros de zona a partir de los 4 landmarks (AM, AL, PM, PL).
    Usa FourZoneComputer para calcular los bboxes y toma sus centros.
    """
    try:
        sys.path.insert(0, str(ROOT))
        from api.four_zone_computer import FourZoneComputer
        zc = FourZoneComputer(landmarks)
        centers = {}
        zone_names = ["AM", "AL", "PM", "PL"]
        for i in range(4):
            x1, y1, x2, y2 = zc.get_zone_bbox(i)
            centers[f"zona_{i}"] = [float((x1 + x2) / 2), float((y1 + y2) / 2)]
        return centers
    except Exception:
        return {f"zona_{i}": [0.0, 0.0] for i in range(4)}


# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
# CONFIGURACIN DE PÁGINA
# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅

st.set_page_config(
    page_title="Hood NN  Análisis de Desgaste Tibial",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Sistema Hood NN  Cuantificación automática de desgaste en bandeja tibial.",
    },
)

# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
# SIDEBAR  ESTADO GLOBAL
# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅

with st.sidebar:
    st.title(" Hood NN")
    st.caption("Análisis de desgaste en bandeja tibial")
    st.divider()

    imgs_count  = len(get_image_list())
    anns        = load_annotations()
    anns_count  = len(anns)
    model_ready = ONNX_MODEL.exists()
    training_on = is_training_running()

    col_m1, col_m2 = st.columns(2)
    col_m1.metric("Imágenes", imgs_count)
    col_m2.metric("Anotadas", anns_count)

    if model_ready:
        st.success(" Modelo ONNX listo")
    else:
        st.warning("️ Sin modelo (entrenar primero)")

    if training_on:
        st.info("⏳ Entrenamiento en curso...")
    else:
        st.caption("Sistema en reposo")

    st.divider()

    # Barra de cobertura de anotaciones
    if imgs_count > 0:
        frac = anns_count / imgs_count
        st.caption(f"Cobertura de anotaciones: {anns_count}/{imgs_count}")
        st.progress(frac)

    st.caption("**Flujo de trabajo:**")
    st.caption("1. ✅ Subir imágenes y anotar")
    st.caption("2. ✅️ Entrenar el modelo")
    st.caption("3. 📤 Exportar a ONNX")
    st.caption("4. ✅ Probar inferencia")


# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
# TABS PRINCIPALES
# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅

tab_datos, tab_train, tab_results, tab_infer = st.tabs([
    "📁 Datos y Anotaciones",
    "✅️ Entrenamiento",
    " Resultados LOOCV",
    "🔬 Probar Modelo",
])


# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
# TAB 1  DATOS Y ANOTACIONES
# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅

with tab_datos:
    st.header("Gestión de imágenes y anotaciones Hood")

    col_left, col_right = st.columns([1, 2], gap="large")

    # ✅ Columna izquierda: subir y listar imágenes ✅✅✅✅✅✅✅✅✅✅✅✅✅✅
    with col_left:
        st.subheader(" Subir imágenes")
        uploaded_files = st.file_uploader(
            "Arrastra archivos JPG/PNG aquí",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="uploader_datos",
        )
        if uploaded_files:
            saved = 0
            for uf in uploaded_files:
                dest = IMAGES_DIR / uf.name
                dest.write_bytes(uf.read())
                saved += 1
            st.success(f" {saved} imagen(es) guardada(s)")
            st.rerun()

        st.divider()
        st.subheader("️ Imágenes en dataset")

        imgs_list   = get_image_list()
        anns_reload = load_annotations()

        if not imgs_list:
            st.info("Sin imágenes todavía. Súbelas arriba.")
        else:
            for img_name in imgs_list:
                annotated = img_name in anns_reload
                status    = "" if annotated else "️"
                st.write(f"{status} {img_name}")

        if st.button("▶ Generar plantilla Excel", width='stretch'):
            import_script = ROOT / "tools" / "import_annotations.py"
            result = subprocess.run(
                [sys.executable, str(import_script), "--template"],
                cwd=str(ROOT), capture_output=True, text=True,
            )
            if result.returncode == 0:
                template_path = ROOT / "tools" / "plantilla_anotaciones.xlsx"
                st.success(f" Plantilla generada: {template_path}")
            else:
                st.error(result.stderr[:400])

    # ✅ Columna derecha: editor de anotaciones ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
    with col_right:
        st.subheader("️ Editor de anotaciones")

        imgs_list = get_image_list()
        if not imgs_list:
            st.warning("Sube imágenes primero.")
        else:
            anns_current = load_annotations()
            selected_img = st.selectbox(
                "Imagen a anotar",
                imgs_list,
                format_func=lambda x: f"{'' if x in anns_current else '️ '} {x}",
            )

            current_ann = anns_current.get(selected_img, _default_annotation(selected_img))

            # ✅ Lateralidad ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
            knee_side_key = f"knee_side_{selected_img}"
            if knee_side_key not in st.session_state:
                st.session_state[knee_side_key] = current_ann.get("knee_side", "derecha")

            img_type_info = _get_image_type(selected_img)
            
            ks_col, ks_hint = st.columns([1, 2])
            with ks_col:
                knee_side = st.radio(
                    "Lateralidad",
                    options=["derecha", "izquierda"],
                    format_func=lambda x: f" Rodilla {x.capitalize()}",
                    index=0 if st.session_state[knee_side_key] == "derecha" else 1,
                    horizontal=True,
                    key=f"radio_knee_{selected_img}",
                )
                st.session_state[knee_side_key] = knee_side
            with ks_hint:
                if img_type_info == "arriba":
                    if knee_side == "derecha":
                        st.info(
                            "Rodilla **derecha** (parte arriba): "
                            "cóndilo **medial**  lado **izquierdo**   \n"
                            "cóndilo **lateral**  lado **derecho** "
                        )
                    else:
                        st.info(
                            "Rodilla **izquierda** (parte arriba): "
                            "cóndilo **medial**  lado **derecho**   \n"
                            "cóndilo **lateral**  lado **izquierdo** "
                        )
                else:
                    st.info(
                        " Imagen de la **parte inferior (_bE)** de la prótesis  \n"
                        "Segmentación en 4 cuadrantes: AM, AL, PM, PL"
                    )

            # ✅ Cargar imagen al inicio para tener dimensiones disponibles ✅
            img_path      = IMAGES_DIR / selected_img
            img_available = img_path.exists()
            img_rgb_orig  = None
            w_orig = h_orig = 0
            if img_available and _CV2_AVAILABLE:
                img_bgr      = cv2_.imread(str(img_path))
                img_rgb_orig = cv2_.cvtColor(img_bgr, cv2_.COLOR_BGR2RGB)
                h_orig, w_orig = img_rgb_orig.shape[:2]

            # ✅ Estado de sesión por imagen ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
            lm_key     = f"lm_coords_{selected_img}"   # zona_0..zona_N centers
            active_key = f"lm_active_{selected_img}"   # int 0-9 o 0-3
            auto_key   = f"lm_auto_{selected_img}"
            img_type_current = _get_image_type(selected_img)

            if lm_key not in st.session_state:
                existing_lms = current_ann.get("landmarks", {})
                if any(any(v != 0 for v in coords) for coords in existing_lms.values()):
                    # Hay landmarks guardados  convertir a centros de zona
                    if img_type_current == "abajo":
                        st.session_state[lm_key] = four_landmarks_to_zone_centers(existing_lms)
                    else:
                        st.session_state[lm_key] = landmarks_to_zone_centers(existing_lms)
                elif img_type_current == "arriba" and w_orig > 0:
                    # Aplicar posiciones por defecto calibradas (fracción  píxeles)
                    side = st.session_state.get(knee_side_key, current_ann.get("knee_side", "derecha"))
                    default_centers = {}
                    for zi, (fx, fy) in enumerate(DEFAULT_ZONE_FRACS_RIGHT):
                        # Espejo horizontal para rodilla izquierda
                        actual_fx = (1.0 - fx) if side == "izquierda" else fx
                        default_centers[f"zona_{zi}"] = [actual_fx * w_orig, fy * h_orig]
                    st.session_state[lm_key] = default_centers
                else:
                    # Inicializar según tipo (abajo o sin imagen cargada)
                    num_zones = 4 if img_type_current == "abajo" else 10
                    st.session_state[lm_key] = {f"zona_{i}": [0.0, 0.0] for i in range(num_zones)}
                st.session_state[auto_key] = False
            if active_key not in st.session_state:
                st.session_state[active_key] = 0

            # ✅ Botón de detección automática ✅✅✅✅✅✅✅✅✅✅✅✅✅✅
            col_auto, col_auto_info = st.columns([1, 3])
            with col_auto:
                run_auto = st.button(
                    "✅ Auto-detectar zonas",
                    key=f"autodetect_{selected_img}",
                    width='stretch',
                    type="primary",
                    disabled=not (img_available and _CV2_AVAILABLE),
                    help="Detecta automáticamente las 10 zonas Hood usando visión computacional. "
                         "Puedes ajustar manualmente después.",
                )
            with col_auto_info:
                if st.session_state.get(auto_key):
                    st.info(
                        "️ Zonas auto-detectadas. Revisa los centros y corrige "
                        "los que no sean correctos haciendo clic sobre la imagen."
                    )
                elif not img_available:
                    st.warning("Sube la imagen primero para habilitar la detección automática.")
                else:
                    st.caption(
                        "Pulsa **Auto-detectar** para obtener la segmentación inicial. "
                        "Luego ajusta cada zona manualmente si es necesario."
                    )

            if run_auto and img_rgb_orig is not None:
                try:
                    sys.path.insert(0, str(ROOT))
                    from api.inference import detect_landmarks_classical
                    detected_lms = detect_landmarks_classical(img_rgb_orig)
                    st.session_state[lm_key]    = landmarks_to_zone_centers(detected_lms)
                    st.session_state[auto_key]  = True
                    st.session_state[active_key] = 0
                    st.rerun()
                except Exception as e_auto:
                    st.error(f"Error en detección automática: {e_auto}")

            zone_centers = st.session_state[lm_key]
            active_idx   = st.session_state[active_key]
            img_type     = _get_image_type(selected_img)
            
            # Definir landmarks según el tipo de imagen
            if img_type == "abajo":
                landmark_labels = {
                    "zona_0": "0 · AM (Anteromedial)",
                    "zona_1": "1 · AL (Anterolateral)",
                    "zona_2": "2 · PM (Posteromedial)",
                    "zona_3": "3 · PL (Posterolateral)",
                }
                max_zones = 4
            else:
                landmark_labels = ZONE_CLICK_LABELS
                max_zones = 10

            # Reiniciar active_idx si está fuera del rango válido
            if active_idx >= max_zones:
                active_idx = 0
                st.session_state[active_key] = 0

            active_zone  = f"zona_{active_idx}"

            # ✅ Botones de selección de zona ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
            st.write(f"**✅ Puntos de segmentación**  pulsa el punto que quieres marcar, luego haz clic en su ubicación:")
            if max_zones == 4:
                # Para 4 puntos: mostrar en 22
                for row_start in [0, 2]:
                    btn_cols = st.columns(2)
                    for col_i, zone_i in enumerate(range(row_start, row_start + 2)):
                        zn        = f"zona_{zone_i}"
                        coords    = zone_centers.get(zn, [0.0, 0.0])
                        has_coord = any(v != 0 for v in coords)
                        with btn_cols[col_i]:
                            if st.button(
                                f"{'' if has_coord else ''} {zone_i}",
                                key=f"lmbtn_{zn}_{selected_img}",
                                type="primary" if zone_i == active_idx else "secondary",
                                width='stretch',
                                help=landmark_labels[zn],
                            ):
                                st.session_state[active_key] = zone_i
                                st.rerun()
            else:
                # Para 10 zonas: mostrar en 2 filas de 5
                for row_start in [0, 5]:
                    btn_cols = st.columns(5)
                    for col_i, zone_i in enumerate(range(row_start, row_start + 5)):
                        zn        = f"zona_{zone_i}"
                        coords    = zone_centers.get(zn, [0.0, 0.0])
                        has_coord = any(v != 0 for v in coords)
                        with btn_cols[col_i]:
                            if st.button(
                                f"{'' if has_coord else ''} {zone_i}",
                                key=f"lmbtn_{zn}_{selected_img}",
                                type="primary" if zone_i == active_idx else "secondary",
                                width='stretch',
                                help=ZONE_CLICK_LABELS[zn],
                            ):
                                st.session_state[active_key] = zone_i
                                st.rerun()

            c_info, c_reset = st.columns([4, 1])
            if img_type == "abajo":
                c_info.info(f" Marcando: **{landmark_labels[active_zone]}")
            else:
                c_info.info(f" Marcando: **{active_zone}**  {ZONE_CLICK_LABELS[active_zone]}")
            if c_reset.button(" Reset", key=f"reset_lm_{selected_img}", width='stretch'):
                st.session_state[lm_key]    = {f"zona_{i}": [0.0, 0.0] for i in range(max_zones)}
                st.session_state[active_key] = 0
                st.rerun()

            # ✅ Imagen interactiva  Vista de zonas ✅✅✅✅✅✅✅✅✅✅✅
            if img_available and _CV2_AVAILABLE:
                DISP_W = 320
                scale  = DISP_W / w_orig
                disp_h = int(h_orig * scale)
                img_disp = cv2_.resize(img_rgb_orig, (DISP_W, disp_h))

                # Dibujar centros de zona ya marcados sobre la imagen de clic
                img_ann = img_disp.copy()
                for zi in range(10):
                    zn     = f"zona_{zi}"
                    coords = zone_centers.get(zn, [0.0, 0.0])
                    if any(v != 0 for v in coords):
                        px_d = int(coords[0] * scale)
                        py_d = int(coords[1] * scale)
                        col  = ZONE_CV_COLORS[zi]
                        r    = 14 if zi == active_idx else 10
                        cv2_.circle(img_ann, (px_d, py_d), r + 2, (255, 255, 255), -1)
                        cv2_.circle(img_ann, (px_d, py_d), r, col, -1)
                        num_str = str(zi)
                        (tw, th), _ = cv2_.getTextSize(
                            num_str, cv2_.FONT_HERSHEY_SIMPLEX, 0.42, 1
                        )
                        cv2_.putText(
                            img_ann, num_str,
                            (px_d - tw // 2, py_d + th // 2),
                            cv2_.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2_.LINE_AA,
                        )

                col_click, col_zones = st.columns(2, gap="small")

                # ✅ Panel izquierdo: imagen clicable ✅✅✅✅✅✅✅✅✅✅
                with col_click:
                    st.caption("️ Haz clic en el **centro** de la zona seleccionada")
                    if _SIC_AVAILABLE:
                        click_val = sic(
                            PILImage.fromarray(img_ann),
                            key=f"imgclick_{selected_img}_{active_idx}",
                        )
                        if click_val is not None:
                            ox = click_val["x"] / scale
                            oy = click_val["y"] / scale
                            st.session_state[lm_key][active_zone] = [ox, oy]
                            if active_idx < 9:
                                st.session_state[active_key] = active_idx + 1
                            st.rerun()
                    else:
                        st.image(img_ann, width='stretch')
                        st.warning("`streamlit-image-coordinates` no instalado. `pip install streamlit-image-coordinates`")
                        coords_cur = zone_centers.get(active_zone, [0.0, 0.0])
                        ni_c1, ni_c2 = st.columns(2)
                        nx_ = ni_c1.number_input("X", value=float(coords_cur[0]), step=1.0,
                            key=f"ni_x_{active_zone}_{selected_img}")
                        ny_ = ni_c2.number_input("Y", value=float(coords_cur[1]), step=1.0,
                            key=f"ni_y_{active_zone}_{selected_img}")
                        if st.button(f"Fijar zona {active_idx}",
                            key=f"ni_save_{active_zone}_{selected_img}",
                            type="primary", width='stretch'):
                            st.session_state[lm_key][active_zone] = [nx_, ny_]
                            if active_idx < 9:
                                st.session_state[active_key] = active_idx + 1
                            st.rerun()

                # ✅ Panel derecho: visualización de zonas ✅✅✅✅✅
                with col_zones:
                    img_type_vis = _get_image_type(selected_img)
                    max_zones_vis = 4 if img_type_vis == "abajo" else 10
                    min_zones_to_show = 2 if img_type_vis == "abajo" else 4
                    
                    n_placed = sum(
                        1 for i in range(max_zones_vis)
                        if any(v != 0 for v in zone_centers.get(f"zona_{i}", [0, 0]))
                    )
                    if n_placed >= min_zones_to_show:
                        st.caption("️ Zonas segmentadas (vista previa)")
                        try:
                            if img_type_vis == "abajo":
                                from api.four_zone_computer import FourZoneComputer
                                # Convertir zona_X a nombres de landmarks (AM, AL, PM, PL)
                                four_zone_names = {
                                    "zona_0": "AM",
                                    "zona_1": "AL", 
                                    "zona_2": "PM",
                                    "zona_3": "PL",
                                }
                                four_lms = {
                                    four_zone_names[f"zona_{i}"]: zone_centers.get(f"zona_{i}", [0, 0])
                                    for i in range(4)
                                }
                                zc = FourZoneComputer(four_lms, image_rgb=img_rgb_orig)
                            else:
                                from api.zone_computer import HoodZoneComputer
                                derived_lms = zone_centers_to_landmarks(zone_centers, w_orig, h_orig)
                                zc = HoodZoneComputer(derived_lms)
                            
                            vis = zc.draw_zones(img_rgb_orig)
                            # Superponer centros de zona cliqueados
                            for zi in range(max_zones_vis):
                                coords = zone_centers.get(f"zona_{zi}", [0.0, 0.0])
                                if any(v != 0 for v in coords):
                                    cx_ = int(coords[0]); cy_ = int(coords[1])
                                    cv2_.circle(vis, (cx_, cy_), 5, (255, 255, 255), -1)
                                    col = (255, 0, 0) if zi < max_zones_vis else (100, 100, 100)
                                    cv2_.circle(vis, (cx_, cy_), 3, col, -1)
                            vis_disp = cv2_.resize(vis, (DISP_W, disp_h))
                            if _SIC_AVAILABLE:
                                st.caption(f" Clic para mover **zona {active_idx}**")
                                click_vis = sic(
                                    PILImage.fromarray(vis_disp),
                                    key=f"visclick_{selected_img}_{active_idx}",
                                )
                                if click_vis is not None:
                                    ox = click_vis["x"] / scale
                                    oy = click_vis["y"] / scale
                                    st.session_state[lm_key][active_zone] = [ox, oy]
                                    if active_idx < max_zones_vis - 1:
                                        st.session_state[active_key] = active_idx + 1
                                    st.rerun()
                            else:
                                st.image(vis_disp, width='stretch')
                            st.caption(
                                f"{n_placed}/{max_zones_vis} zonas marcadas · "
                                f"{'Segmentación completa ' if n_placed == max_zones_vis else 'Segmentación parcial ️'}"
                            )
                        except Exception as ez:
                            st.warning(f"No se puede mostrar zonas: {ez}")
                            st.image(img_disp, width='stretch')
                    else:
                        st.caption("️ Zonas segmentadas")
                        st.image(img_disp, width='stretch')
                        st.info(
                            f"Marca {min_zones_to_show - n_placed} zona(s) más para ver la segmentación. "
                            f"Ya tienes: {n_placed}/{max_zones_vis}"
                        )

            elif not img_available:
                st.warning(
                    f"Imagen `{selected_img}` no encontrada en `data/images/`. "
                    "Súbela en la columna izquierda."
                )
            else:
                st.warning("OpenCV no disponible: `pip install opencv-python`")

            # Calcular número de zonas marcadas
            img_type_crop = _get_image_type(selected_img)
            max_zones_crop = 4 if img_type_crop == "abajo" else 10
            n_placed = sum(
                1 for i in range(max_zones_crop)
                if any(v != 0 for v in zone_centers.get(f"zona_{i}", [0, 0]))
            )

            # Derivar landmarks anatómicos desde zone centers para guardar en annotations.json
            if w_orig > 0:
                if img_type_crop == "abajo":
                    # Para abajo, convertir zona_X a AM, AL, PM, PL
                    four_zone_names = {
                        "zona_0": "AM",
                        "zona_1": "AL",
                        "zona_2": "PM",
                        "zona_3": "PL",
                    }
                    new_lms = {
                        four_zone_names[f"zona_{i}"]: zone_centers.get(f"zona_{i}", [0, 0])
                        for i in range(4)
                    }
                else:
                    new_lms = zone_centers_to_landmarks(zone_centers, w_orig, h_orig)
            else:
                if img_type_crop == "abajo":
                    new_lms = {lm: [0.0, 0.0] for lm in ["AM", "AL", "PM", "PL"]}
                else:
                    new_lms = {f"zona_{i}": [0.0, 0.0] for i in range(max_zones_crop)}

            # ✅ Vista de recortes por zona ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
            if img_rgb_orig is not None and n_placed >= 2:
                try:
                    if img_type_crop == "abajo":
                        from api.four_zone_computer import FourZoneComputer
                        # Convertir zona_X a nombres de landmarks (AM, AL, PM, PL)
                        four_zone_names = {
                            "zona_0": "AM",
                            "zona_1": "AL",
                            "zona_2": "PM",
                            "zona_3": "PL",
                        }
                        four_lms = {
                            four_zone_names[f"zona_{i}"]: zone_centers.get(f"zona_{i}", [0, 0])
                            for i in range(4)
                        }
                        zc_crops = FourZoneComputer(four_lms, image_rgb=img_rgb_orig)
                        zone_names_crops = ["AM", "AL", "PM", "PL"]
                        with st.expander("✅ Recortes de cada zona (AM, AL, PM, PL)", expanded=(n_placed == 4)):
                            crop_cols = st.columns(2)
                            for zi in range(4):
                                crop = zc_crops.get_zone_crop(img_rgb_orig, zi, output_size=320)
                                # Guardar recorte en data/crops/ (persistente entre sesiones)
                                crop_path = CROPS_DIR / f"{Path(selected_img).stem}_zona_{zi}_{zone_names_crops[zi]}.png"
                                try:
                                    import cv2 as _cv2
                                    _cv2.imwrite(str(crop_path), _cv2.cvtColor(crop, _cv2.COLOR_RGB2BGR))
                                except Exception:
                                    pass
                                with crop_cols[zi % 2]:
                                    st.image(
                                        crop,
                                        caption=f"Zona {zi}  {zone_names_crops[zi]}",
                                        width='stretch',
                                    )
                    else:
                        from api.zone_computer import HoodZoneComputer
                        derived_lms_crops = zone_centers_to_landmarks(zone_centers, w_orig, h_orig)
                        zc_crops = HoodZoneComputer(derived_lms_crops)
                        with st.expander("✅ Recortes de cada zona (0  9)", expanded=(n_placed == 10)):
                            for row_start in (0, 5):
                                crop_cols = st.columns(5)
                                for zi in range(row_start, row_start + 5):
                                    crop = zc_crops.get_zone_crop(img_rgb_orig, zi, output_size=320)
                                    # Guardar recorte en data/crops/ (persistente entre sesiones)
                                    crop_path = CROPS_DIR / f"{Path(selected_img).stem}_zona_{zi}_{ZONE_NAMES[zi]}.png"
                                    try:
                                        import cv2 as _cv2
                                        _cv2.imwrite(str(crop_path), _cv2.cvtColor(crop, _cv2.COLOR_RGB2BGR))
                                    except Exception:
                                        pass
                                    with crop_cols[zi - row_start]:
                                        st.image(
                                            crop,
                                            caption=f"Zona {zi}  {ZONE_NAMES[zi]}",
                                            width='stretch',
                                        )
                except Exception as e_crop:
                    st.caption(f"No se pueden mostrar recortes: {e_crop}")

            # ✅ Scores Hood (tabla editable) ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
            st.write("**Scores de daño**  0 = sin daño · 1 = <10% · 2 = 10-50% · 3 = >50%")

            score_data = current_ann.get("damage_scores", {})
            img_type   = _get_image_type(selected_img)

            # Detectar número de zonas dinámicamente
            if img_type == "abajo":
                num_zones = 4
                zone_labels = ["AM", "AL", "PM", "PL"]
            else:
                num_zones = 10
                zone_labels = ZONE_NAMES

            # Construir DataFrame dinámico
            scores_matrix = []
            for z in range(num_zones):
                row = score_data.get(f"zona_{z}", [0] * 7)
                scores_matrix.append([int(v) for v in row[:7]])

            df_scores = pd.DataFrame(
                scores_matrix,
                index=zone_labels,
                columns=DAMAGE_TYPES,
            )

            # Tabla editable con selectboxes de 0-3 por celda
            edited_df = st.data_editor(
                df_scores,
                column_config={
                    dmg: st.column_config.SelectboxColumn(
                        dmg.capitalize(),
                        options=[0, 1, 2, 3],
                        required=True,
                        width="small",
                    )
                    for dmg in DAMAGE_TYPES
                },
                width='stretch',
                height=(180 if num_zones == 4 else 420),
                key=f"scores_editor_{selected_img}",
            )

            # Score total calculado en tiempo real
            total_score = int(edited_df.values.sum())
            max_score   = 3 * 7 * num_zones  # 3 (máx)  7 daños  número zonas
            pct         = total_score / max_score * 100
            col_sc1, col_sc2, col_sc3 = st.columns(3)
            col_sc1.metric("Score total", f"{total_score} / {max_score}")
            col_sc2.metric("Porcentaje de daño", f"{pct:.1f}%")
            col_sc3.metric(
                "Nivel de daño",
                "Severo" if pct > 50 else "Moderado" if pct > 20 else "Leve" if pct > 5 else "Mínimo",
            )

            if st.button(" Guardar anotaciones", type="primary", width='stretch'):
                anns_latest = load_annotations()
                new_scores  = {
                    f"zona_{z}": edited_df.iloc[z].tolist()
                    for z in range(num_zones)
                }
                anns_latest[selected_img] = {
                    "knee_side":     st.session_state.get(knee_side_key, "derecha"),
                    "image_type":    img_type,
                    "landmarks":     new_lms,
                    "damage_scores": new_scores,
                }
                save_annotations(anns_latest)
                st.success(f" Guardado: {selected_img}  (score {total_score}/{max_score})")


# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
# TAB 2  ENTRENAMIENTO
# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅

with tab_train:
    st.header("Entrenamiento del modelo HoodNet")

    anns_reload  = load_annotations()
    anns_n       = len(anns_reload)
    training_now = is_training_running()

    if anns_n == 0:
        st.error(" No hay imágenes anotadas. Ve a la pestaña 📁 Datos primero.")
    elif anns_n < 3:
        st.warning(
            f"️ Solo {anns_n} imagen(es) anotada(s). "
            "Se recomiendan al menos 5-10 para obtener resultados significativos. "
            "Puedes continuar de todos modos."
        )

    # ✅ Previsualización de Data Augmentation ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
    with st.expander(" Previsualizar Data Augmentation", expanded=False):
        aug_imgs_avail = get_image_list()
        aug_imgs_aE    = [f for f in aug_imgs_avail if f.endswith("_aE.png")]
        if not aug_imgs_aE:
            st.info("No hay imágenes _aE disponibles para previsualizar.")
        else:
            _AUG_ZONE_NAMES = [
                "0 · Med periférico", "1 · Med anterior",  "2 · Med central",   "3 · Med posterior",
                "4 · Lat central",    "5 · Lat anterior",  "6 · Lat periférico", "7 · Lat posterior",
                "8 · Surco anterior", "9 · Surco posterior",
            ]
            col_aug_sel, col_aug_zone, col_aug_n, col_aug_btn = st.columns([2, 2, 1, 1])
            aug_selected = col_aug_sel.selectbox(
                "Imagen base", aug_imgs_aE, key="aug_preview_img",
            )
            aug_zone_label = col_aug_zone.selectbox(
                "Zona Hood", _AUG_ZONE_NAMES, key="aug_preview_zone",
            )
            aug_zone_idx = _AUG_ZONE_NAMES.index(aug_zone_label)
            aug_n = col_aug_n.slider(
                "Nº variantes", min_value=4, max_value=16, value=8, step=4,
                key="aug_preview_n",
            )
            run_aug_preview = col_aug_btn.button(
                "▶ Generar", type="primary",
                width='stretch', key="aug_preview_btn",
            )

            if run_aug_preview:
                if not _CV2_AVAILABLE:
                    st.error("OpenCV no disponible (`pip install opencv-python`).")
                else:
                    aug_path = IMAGES_DIR / aug_selected
                    img_bgr  = cv2_.imread(str(aug_path))
                    if img_bgr is None:
                        st.error(f"No se pudo cargar {aug_selected}")
                    else:
                        img_rgb = cv2_.cvtColor(img_bgr, cv2_.COLOR_BGR2RGB)
                        # Extraer recorte de zona antes de augmentar
                        _ann_for_aug = load_annotations().get(aug_selected, {})
                        _lm_for_aug  = _ann_for_aug.get("landmarks", {})
                        _crop_ok     = False
                        if _lm_for_aug and len(_lm_for_aug) >= 7:
                            try:
                                sys.path.insert(0, str(ROOT))
                                from api.zone_computer import HoodZoneComputer as _HZC
                                _zc_aug  = _HZC(_lm_for_aug)
                                img_rgb  = _zc_aug.get_zone_crop(img_rgb, aug_zone_idx, output_size=None)
                                if img_rgb is not None and img_rgb.size > 0 and min(img_rgb.shape[:2]) > 0:
                                    _crop_ok = True
                            except Exception as _e_crop:
                                st.warning(f"No se pudo extraer el recorte de zona: {_e_crop}. "
                                           "Se usará la imagen completa.")
                        if not _crop_ok:
                            st.warning("Sin landmarks para esta imagen  se augmenta la imagen completa.")
                            img_rgb = cv2_.cvtColor(cv2_.imread(str(aug_path)), cv2_.COLOR_BGR2RGB)
                        try:
                            # Asegurar que el user site-packages está en sys.path
                            import site as _site
                            _usp = _site.getusersitepackages()
                            if _usp not in sys.path:
                                sys.path.insert(0, _usp)
                            import albumentations as _A
                            from albumentations.pytorch import ToTensorV2 as _ToTensorV2

                            _IMG = 320
                            _transform = _A.Compose([
                                _A.Resize(_IMG, _IMG),
                                _A.HorizontalFlip(p=0.4),
                                _A.VerticalFlip(p=0.1),
                                _A.Rotate(limit=30, p=0.8),
                                _A.ShiftScaleRotate(
                                    shift_limit=0.08, scale_limit=0.15,
                                    rotate_limit=0, p=0.6,
                                ),
                                _A.Perspective(scale=(0.02, 0.06), p=0.4),
                                _A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.3),
                                _A.OpticalDistortion(distort_limit=0.1, p=0.2),
                                _A.CoarseDropout(
                                    num_holes_range=(1, 6),
                                    hole_height_range=(0.05, 0.10),
                                    hole_width_range=(0.05, 0.10),
                                    fill=0, p=0.3,
                                ),
                                _A.RandomBrightnessContrast(0.35, 0.35, p=0.8),
                                _A.RandomGamma(gamma_limit=(70, 140), p=0.4),
                                _A.HueSaturationValue(8, 30, 30, p=0.6),
                                _A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
                                _A.ToGray(p=0.1),
                                _A.GaussNoise(std_range=(0.04, 0.20), p=0.4),
                                _A.SaltAndPepper(amount=(0.001, 0.01), p=0.2),
                                _A.OneOf([
                                    _A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                                    _A.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.2), p=1.0),
                                    _A.MotionBlur(blur_limit=5, p=1.0),
                                ], p=0.4),
                                _A.ImageCompression(quality_range=(75, 100), p=0.3),
                                _A.RandomShadow(
                                    shadow_roi=(0, 0, 1, 1),
                                    num_shadows_limit=(1, 2),
                                    shadow_dimension=4, p=0.2,
                                ),
                                # No ToTensorV2  queremos el array NumPy directamente
                            ])

                            with st.spinner(f"Generando {aug_n} variantes..."):
                                aug_results = []
                                for _ in range(aug_n):
                                    out = _transform(image=img_rgb)
                                    aug_results.append(out["image"])

                            st.session_state["aug_preview_results"]  = aug_results
                            st.session_state["aug_preview_base"]     = aug_selected
                            st.session_state["aug_preview_zone_lbl"] = aug_zone_label
                            st.session_state["aug_preview_zone_idx"] = aug_zone_idx
                        except Exception as e_aug:
                            st.error(f"Error generando augmentación: {e_aug}")
                            import traceback; st.code(traceback.format_exc())

            # Mostrar resultados (persisten en sesión hasta nueva generación)
            if "aug_preview_results" in st.session_state:
                results    = st.session_state["aug_preview_results"]
                base_label = st.session_state.get("aug_preview_base", "")
                zone_lbl   = st.session_state.get("aug_preview_zone_lbl", "")
                zone_i     = st.session_state.get("aug_preview_zone_idx", 0)
                st.caption(
                    f"**{len(results)} variantes augmentadas**  `{base_label}` · {zone_lbl} "
                    "(cada click genera combinaciones distintas)"
                )
                # Recorte original de la zona al lado de las variantes
                orig_crop_shown = False
                _ann_disp = load_annotations().get(base_label, {})
                _lm_disp  = _ann_disp.get("landmarks", {})
                if _lm_disp and len(_lm_disp) >= 7 and _CV2_AVAILABLE:
                    try:
                        from api.zone_computer import HoodZoneComputer as _HZC2
                        _orig_bgr  = cv2_.imread(str(IMAGES_DIR / base_label))
                        _orig_rgb  = cv2_.cvtColor(_orig_bgr, cv2_.COLOR_BGR2RGB)
                        _zc_disp   = _HZC2(_lm_disp)
                        _orig_crop = _zc_disp.get_zone_crop(_orig_rgb, zone_i, output_size=None)
                        if _orig_crop is not None and min(_orig_crop.shape[:2]) > 0:
                            _orig_crop = cv2_.resize(_orig_crop, (320, 320))
                            all_imgs  = [_orig_crop] + results
                            captions  = [f" Zona {zone_i} original"] + [f"Aug {i+1}" for i in range(len(results))]
                            orig_crop_shown = True
                    except Exception:
                        pass
                if not orig_crop_shown:
                    all_imgs = results
                    captions = [f"Aug {i+1}" for i in range(len(results))]

                n_cols = 4
                for row_start in range(0, len(all_imgs), n_cols):
                    row_imgs = all_imgs[row_start:row_start + n_cols]
                    row_caps = captions[row_start:row_start + n_cols]
                    cols_row = st.columns(n_cols)
                    for ci, (img_show, cap) in enumerate(zip(row_imgs, row_caps)):
                        cols_row[ci].image(img_show, caption=cap, width='stretch')

    # ✅ Generación masiva de crops augmentados ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
    with st.expander("▶ Generar todos los crops augmentados", expanded=False):
        st.markdown(
            "Genera recortes augmentados de **todas las imágenes**  **todas las zonas** "
            "y los guarda en `data/augmented_crops/`. "
            "til para inspeccionar el dataset antes de entrenar."
        )
        _col_n_aug, _col_gen_all = st.columns([1, 2])
        _n_aug_all = _col_n_aug.slider(
            "Variantes por zona", min_value=2, max_value=16, value=8, step=2,
            key="gen_all_n",
        )
        _btn_gen_all = _col_gen_all.button(
            "▶ Generar todos los crops", type="primary",
            width='stretch', key="gen_all_btn",
        )
        if _btn_gen_all:
            if not _CV2_AVAILABLE:
                st.error("OpenCV no disponible.")
            else:
                try:
                    import site as _sa; _up = _sa.getusersitepackages()
                    if _up not in sys.path: sys.path.insert(0, _up)
                    import albumentations as _A2
                    from albumentations.pytorch import ToTensorV2 as _TTV2
                    from api.zone_computer import HoodZoneComputer as _HZC3

                    _IMG2 = 224
                    _aug_transform = _A2.Compose([
                        _A2.Resize(_IMG2, _IMG2),
                        _A2.HorizontalFlip(p=0.4),
                        _A2.VerticalFlip(p=0.1),
                        _A2.Rotate(limit=30, p=0.8),
                        _A2.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=0, p=0.6),
                        _A2.Perspective(scale=(0.02, 0.06), p=0.4),
                        _A2.GridDistortion(num_steps=5, distort_limit=0.15, p=0.3),
                        _A2.OpticalDistortion(distort_limit=0.1, p=0.2),
                        _A2.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(0.05, 0.10), hole_width_range=(0.05, 0.10), fill=0, p=0.3),
                        _A2.RandomBrightnessContrast(0.35, 0.35, p=0.8),
                        _A2.RandomGamma(gamma_limit=(70, 140), p=0.4),
                        _A2.HueSaturationValue(8, 30, 30, p=0.6),
                        _A2.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.4),
                        _A2.ToGray(p=0.1),
                        _A2.GaussNoise(std_range=(0.04, 0.20), p=0.4),
                        _A2.SaltAndPepper(amount=(0.001, 0.01), p=0.2),
                        _A2.OneOf([
                            _A2.GaussianBlur(blur_limit=(3, 7), p=1.0),
                            _A2.Sharpen(alpha=(0.1, 0.4), lightness=(0.8, 1.2), p=1.0),
                            _A2.MotionBlur(blur_limit=5, p=1.0),
                        ], p=0.4),
                        _A2.ImageCompression(quality_range=(75, 100), p=0.3),
                        _A2.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 2), shadow_dimension=4, p=0.2),
                    ])

                    _all_anns  = load_annotations()
                    _aE_imgs   = [f for f in _all_anns if f.endswith("_aE.png") and (IMAGES_DIR / f).exists()]
                    _valid_imgs = [
                        (f, _all_anns[f]) for f in _aE_imgs
                        if len(_all_anns[f].get("landmarks", {})) >= 7
                    ]

                    if not _valid_imgs:
                        st.warning("No hay imágenes _aE con landmarks completos.")
                    else:
                        _total_steps = len(_valid_imgs) * 10  # imágenes  zonas
                        _prog        = st.progress(0, text="Iniciando")
                        _step        = 0
                        _saved       = 0
                        _errors      = []

                        for _img_name, _ann in _valid_imgs:
                            _img_bgr = cv2_.imread(str(IMAGES_DIR / _img_name))
                            _img_rgb = cv2_.cvtColor(_img_bgr, cv2_.COLOR_BGR2RGB)
                            _stem    = Path(_img_name).stem
                            try:
                                _zc3 = _HZC3(_ann["landmarks"])
                            except Exception as _ez:
                                _errors.append(f"{_img_name}: {_ez}")
                                _step += 10
                                _prog.progress(min(_step / _total_steps, 1.0))
                                continue

                            for _zi in range(10):
                                _zone_dir = AUG_CROPS_DIR / _stem / f"zona_{_zi}"
                                _zone_dir.mkdir(parents=True, exist_ok=True)
                                try:
                                    _crop = _zc3.get_zone_crop(_img_rgb, _zi, output_size=None)
                                    if _crop is None or _crop.size == 0 or min(_crop.shape[:2]) == 0:
                                        raise ValueError("recorte vacío")
                                    # Guardar el crop original (sin augmentar)
                                    _orig_path = _zone_dir / "original.png"
                                    cv2_.imwrite(str(_orig_path), cv2_.cvtColor(_crop, cv2_.COLOR_RGB2BGR))
                                    _saved += 1
                                    # Generar N variantes augmentadas
                                    for _k in range(_n_aug_all):
                                        _out = _aug_transform(image=_crop)
                                        _aug_img = _out["image"]
                                        _aug_bgr = cv2_.cvtColor(_aug_img, cv2_.COLOR_RGB2BGR)
                                        cv2_.imwrite(
                                            str(_zone_dir / f"aug_{_k:02d}.png"),
                                            _aug_bgr,
                                        )
                                        _saved += 1
                                except Exception as _ez2:
                                    _errors.append(f"{_img_name} zona {_zi}: {_ez2}")

                                _step += 1
                                _prog.progress(
                                    min(_step / _total_steps, 1.0),
                                    text=f"Procesando {_img_name} · zona {_zi}/9",
                                )

                        _prog.progress(1.0, text=" Completado")
                        st.success(
                            f"**{_saved}** imágenes guardadas en `data/augmented_crops/` "
                            f"({len(_valid_imgs)} muestras  10 zonas  {_n_aug_all} variantes + original)."
                        )
                        if _errors:
                            st.warning(f"{len(_errors)} advertencias:\n" + "\n".join(_errors[:10]))
                except Exception as _e_all:
                    st.error(f"Error: {_e_all}")
                    import traceback; st.code(traceback.format_exc())

        # Mostrar estadísticas de lo que hay guardado
        if AUG_CROPS_DIR.exists():
            _aug_subdirs = [d for d in AUG_CROPS_DIR.iterdir() if d.is_dir()]
            if _aug_subdirs:
                _total_files = sum(len(list(d.rglob("*.png"))) for d in _aug_subdirs)
                st.caption(
                    f"✅ `data/augmented_crops/`: **{len(_aug_subdirs)}** imágenes · "
                    f"**{_total_files}** archivos guardados"
                )

    col_cfg, col_ctrl = st.columns([1, 2], gap="large")

    # ✅ Configuración ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
    with col_cfg:
        st.subheader("️ Configuración")

        epochs = st.slider(
            "pocas de entrenamiento", min_value=10, max_value=120, value=60, step=5,
            help="Número de pasadas completas sobre el dataset por zona. "
                 "Más épocas = más entrenamiento. En CPU 60 épocas  5 min.",
        )
        skip_loocv = st.checkbox(
            "Solo modelo final (sin LOOCV)",
            value=False,
            help="LOOCV entrena N modelos (uno por imagen dejada fuera). "
                 "Desactiva para entrenamiento más rápido sin métricas de validación.",
        )
        dry_run = st.checkbox(
            "Dry-run (solo 2 épocas  verificar que funciona)",
            value=False,
            help="Ejecuta 2 épocas y 1 fold para verificar que el pipeline no tiene errores.",
        )

        # Estimación de tiempo
        n_folds  = 1 if skip_loocv else anns_n
        ep_final = 2 if dry_run else epochs
        mins_est = max(1, (n_folds * 10 * ep_final) // 60)

        st.info(
            f"**Estimación en CPU:**  \n"
            f"{'Dry-run: ~1 min' if dry_run else f'~{mins_est} min'}"
            f"{'  (LOOCV: ' + str(n_folds) + ' folds)' if not skip_loocv and not dry_run else ''}"
        )

    # ✅ Control ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
    with col_ctrl:
        st.subheader("️ Control")

        can_resume = has_checkpoint() and not training_now

        col_b1, col_b2, col_b3, col_b4 = st.columns(4)

        # ── Iniciar (desde cero) ────────────────────────────────────────────
        def _launch_training(cmd, append_log=False):
            """Lanza train.py como subproceso y guarda el PID."""
            log_mode = "a" if append_log else "w"
            log_fd   = open(LOG_FILE, log_mode, encoding="utf-8")
            import site as _site2
            _env = os.environ.copy()
            _usp = _site2.getusersitepackages()
            _existing = _env.get("PYTHONPATH", "")
            _env["PYTHONPATH"] = _usp + (os.pathsep + _existing if _existing else "")
            _env["PYTHONUNBUFFERED"]  = "1"
            _env["PYTHONIOENCODING"] = "utf-8"
            _cflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
            proc = subprocess.Popen(
                cmd, cwd=str(ROOT),
                stdout=log_fd, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True, env=_env, creationflags=_cflags,
            )
            log_fd.close()
            PID_FILE.write_text(str(proc.pid))
            return proc.pid

        with col_b1:
            if st.button(
                "▶ Iniciar",
                type="primary",
                disabled=training_now or anns_n == 0,
                width='stretch',
                help="Inicia el entrenamiento desde cero (borra cualquier checkpoint previo)",
            ):
                train_script = ROOT / "train" / "train.py"
                cmd = [
                    sys.executable, str(train_script),
                    "--epochs",          str(epochs),
                    "--progress-file",   str(PROGRESS_FILE),
                    "--checkpoint-file", str(CHECKPOINT_FILE),
                    "--stop-file",       str(STOP_FILE),
                ]
                if skip_loocv:
                    cmd.append("--skip-loocv")
                if dry_run:
                    cmd.append("--dry-run")

                # Limpiar archivos de sesión anterior (inicio limpio)
                for f_ in [PROGRESS_FILE, LOG_FILE, CHECKPOINT_FILE, STOP_FILE]:
                    try:
                        f_.unlink(missing_ok=True)
                    except Exception:
                        pass

                pid = _launch_training(cmd, append_log=False)
                st.success(f"Lanzado (PID {pid})")
                st.rerun()

        # ── Reanudar ────────────────────────────────────────────────────────
        with col_b2:
            if st.button(
                "⏩ Reanudar",
                type="primary",
                disabled=training_now or not can_resume,
                width='stretch',
                help="Continúa el entrenamiento desde donde se pausó",
            ):
                # Leer épocas y configuración guardadas en el checkpoint
                try:
                    import torch as _torch
                    _ckpt = _torch.load(str(CHECKPOINT_FILE), weights_only=False)
                    _ckpt_epochs     = _ckpt.get("epochs", epochs)
                    _ckpt_skip_loocv = (_ckpt.get("phase") == "final")
                except Exception:
                    _ckpt_epochs     = epochs
                    _ckpt_skip_loocv = skip_loocv

                train_script = ROOT / "train" / "train.py"
                cmd = [
                    sys.executable, str(train_script),
                    "--resume",
                    "--epochs",          str(_ckpt_epochs),
                    "--progress-file",   str(PROGRESS_FILE),
                    "--checkpoint-file", str(CHECKPOINT_FILE),
                    "--stop-file",       str(STOP_FILE),
                ]
                if _ckpt_skip_loocv:
                    cmd.append("--skip-loocv")

                # Limpiar flag de parada anterior
                STOP_FILE.unlink(missing_ok=True)

                pid = _launch_training(cmd, append_log=True)
                st.success(f"Reanudado (PID {pid})")
                st.rerun()

        # ── Pausar (graceful) ───────────────────────────────────────────────
        with col_b3:
            if st.button(
                "⏸ Pausar",
                type="secondary",
                disabled=not training_now,
                width='stretch',
                help="Guarda el estado y para al terminar la época actual",
            ):
                STOP_FILE.write_text("stop")
                st.warning("Pausa solicitada — el entrenamiento guardará el estado "
                           "al terminar la época actual.")
                time.sleep(1)
                st.rerun()

        # ── Exportar ONNX ───────────────────────────────────────────────────
        with col_b4:
            if st.button(
                "📤 ONNX",
                disabled=not PT_MODEL.exists() or training_now,
                width='stretch',
                help="Convierte hoodnet_final.pt  hood_model.onnx para producción",
            ):
                export_script = ROOT / "train" / "export_onnx.py"
                with st.spinner("Exportando..."):
                    result = subprocess.run(
                        [sys.executable, str(export_script)],
                        cwd=str(ROOT), capture_output=True, text=True,
                    )
                if result.returncode == 0:
                    st.success(" Exportado: models/hood_model.onnx")
                else:
                    st.error(f"Error:\n{result.stderr[:500]}")
                st.rerun()

        # ✅ Progreso ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
        st.divider()

        progress  = read_progress()
        status    = progress.get("status", "")
        phase     = progress.get("phase", "")
        cur_ep    = progress.get("current_epoch", 0)
        tot_ep    = progress.get("total_epochs", epochs)
        cur_fold  = progress.get("current_fold", 0)
        tot_folds = progress.get("total_folds", 1)
        cur_zone  = progress.get("current_zone", 0)
        loss_val  = progress.get("loss", None)

        if status == "done":
            st.success(" ¡Entrenamiento completado! Puedes exportar a ONNX.")
            if PT_MODEL.exists():
                st.caption(f"Modelo guardado: {PT_MODEL.name}")

        elif status == "paused":
            if phase == "LOOCV":
                pct   = (cur_fold - 1 + (cur_ep / max(tot_ep, 1))) / max(tot_folds, 1)
                label = (f"PAUSED · LOOCV  Fold {cur_fold}/{tot_folds}  · "
                         f"  Época {cur_ep}/{tot_ep}  ·  Zona {cur_zone}/10")
            else:
                pct   = cur_ep / max(tot_ep, 1)
                label = f"PAUSED · Modelo final  Época {cur_ep}/{tot_ep}"
            st.progress(min(pct, 1.0), text=label)
            st.info("⏸ Entrenamiento pausado — pulsa **⏩ Reanudar** para continuar "
                    "desde donde se paró.")

        elif status == "training":
            if phase == "LOOCV":
                fold_pct = (cur_fold - 1 + (cur_ep / max(tot_ep, 1))) / max(tot_folds, 1)
                label    = f"LOOCV  Fold {cur_fold}/{tot_folds}  ·  Época {cur_ep}/{tot_ep}  ·  Zona {cur_zone}/10"
            else:
                fold_pct = cur_ep / max(tot_ep, 1)
                label    = f"Modelo final  Época {cur_ep}/{tot_ep}"

            st.progress(min(fold_pct, 1.0), text=label)

            # ── Velocidad y ETA ────────────────────────────────────────────
            epoch_dur = progress.get("epoch_duration_s")
            eta_s     = progress.get("eta_s")
            _met_cols = st.columns(3)
            if epoch_dur is not None:
                speed_label = (f"{epoch_dur:.1f} s/época"
                               if epoch_dur >= 1
                               else f"{epoch_dur * 1000:.0f} ms/época")
                _met_cols[0].metric("⚡ Velocidad", speed_label)
            if eta_s is not None:
                _met_cols[1].metric("⏳ Tiempo restante", _fmt_eta(eta_s))
            if loss_val is not None:
                _met_cols[2].metric("📉 Pérdida", f"{loss_val:.4f}")

        elif status == "starting":
            st.info("⏳ Iniciando...")

        elif training_now:
            st.info("⏳ Entrenamiento en curso (esperando primer update)...")

        # ✅ Log en tiempo real ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
        if LOG_FILE.exists() or training_now:
            with st.expander(" Log de entrenamiento", expanded=training_now):
                st.code(read_log_tail(30), language="text")

        # Auto-rerun cada 8 segundos mientras entrena
        if training_now:
            time.sleep(8)
            st.rerun()


# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
# TAB 3  RESULTADOS LOOCV
# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅

with tab_results:
    st.header("Resultados de validación cruzada (LOOCV)")

    if not LOOCV_FILE.exists():
        st.info(
            "Todavía no hay resultados LOOCV. "
            "Completa un entrenamiento con LOOCV activado."
        )
    else:
        with open(LOOCV_FILE, encoding="utf-8") as f:
            loocv_data = json.load(f)

        st.success(f" {len(loocv_data)} folds disponibles")

        # ✅ Tabla resumen ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
        rows = []
        for fold_results in loocv_data:
            for zone_key, metrics in fold_results.items():
                mae_per = metrics.get("mae_per_damage", {})
                row = {
                    "zona": zone_key,
                    "exact_match": metrics.get("exact_match", 0.0),
                    "mae_global": mae_per.get("_global", 0.0),
                }
                for dmg in DAMAGE_TYPES:
                    row[f"mae_{dmg}"] = mae_per.get(dmg, 0.0)
                rows.append(row)

        if rows:
            df_all = pd.DataFrame(rows)
            summary = df_all.groupby("zona").agg(
                EM_media=("exact_match", "mean"),
                EM_std=("exact_match", "std"),
                MAE_media=("mae_global", "mean"),
                MAE_std=("mae_global", "std"),
            ).round(3)

            # Renombrar índice para visualización
            summary.index = [z.replace("_", " ") for z in summary.index]

            col_t1, col_t2 = st.columns([2, 1])
            with col_t1:
                st.subheader("Métricas por zona")
                st.dataframe(summary, width='stretch')
            with col_t2:
                global_em  = df_all["exact_match"].mean()
                global_mae = df_all["mae_global"].mean()
                st.metric("Exact Match Global", f"{global_em:.3f}")
                st.metric("MAE Global", f"{global_mae:.3f}")
                st.caption(
                    "**Exact Match**: fracción de muestras donde "
                    "los 8 scores son exactamente correctos."
                )
                st.caption(
                    "**MAE**: error ordinal medio (|predicho  real|) "
                    "en escala 0-3."
                )

            if _PLOTLY:
                st.divider()
                col_g1, col_g2 = st.columns(2)

                with col_g1:
                    df_plot = summary.reset_index().rename(columns={"zona": "Zona"})
                    fig1 = px.bar(
                        df_plot, x="Zona", y="EM_media",
                        error_y="EM_std",
                        title="Exact Match por zona (media ± std)",
                        labels={"EM_media": "Exact Match"},
                        color="EM_media",
                        color_continuous_scale="RdYlGn",
                        range_color=[0, 1],
                    )
                    fig1.update_xaxes(tickangle=40)
                    fig1.update_layout(showlegend=False)
                    st.plotly_chart(fig1, width='stretch')

                with col_g2:
                    fig2 = px.bar(
                        df_plot, x="Zona", y="MAE_media",
                        error_y="MAE_std",
                        title="MAE global por zona (media ± std)",
                        labels={"MAE_media": "MAE"},
                        color="MAE_media",
                        color_continuous_scale="RdYlGn_r",
                        range_color=[0, 1.5],
                    )
                    fig2.update_xaxes(tickangle=40)
                    fig2.update_layout(showlegend=False)
                    st.plotly_chart(fig2, width='stretch')

                # Heatmap MAE por zona  tipo de daño
                st.subheader("MAE por zona y tipo de daño")
                mae_cols  = [f"mae_{d}" for d in DAMAGE_TYPES]
                df_heatmap = df_all.groupby("zona")[mae_cols].mean()
                df_heatmap.columns = DAMAGE_TYPES
                df_heatmap.index   = [z.replace("_", " ") for z in df_heatmap.index]

                fig_hm = px.imshow(
                    df_heatmap,
                    color_continuous_scale=["#f5f5f5", "#fff176", "#ffb74d", "#ef5350"],
                    zmin=0, zmax=2,
                    text_auto=".2f",
                    title="MAE medio por zona  tipo de daño",
                    aspect="auto",
                )
                st.plotly_chart(fig_hm, width='stretch')


# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅
# TAB 4  PROBAR MODELO
# ✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅✅

with tab_infer:
    st.header("Probar el modelo con una imagen nueva")

    if not ONNX_MODEL.exists():
        st.warning(
            "️ No hay modelo ONNX disponible.  \n"
            "Completa el entrenamiento (pestaña ✅️) y exporta el modelo "
            "con el botón **📤 Exportar a ONNX**."
        )
        st.stop()

    st.success(f" Modelo listo: `{ONNX_MODEL.name}`")

    infer_img = st.file_uploader(
        "Sube una imagen de bandeja tibial para analizar",
        type=["jpg", "jpeg", "png"],
        key="infer_upload",
    )

    if not infer_img:
        st.info("Sube una imagen para comenzar el análisis.")
    else:
        try:
            from PIL import Image as PILImage
            pil_img   = PILImage.open(infer_img).convert("RGB")
            img_array = np.array(pil_img, dtype=np.uint8)
        except Exception as ex:
            st.error(f"No se pudo leer la imagen: {ex}")
            st.stop()

        col_img, col_res = st.columns([1, 2], gap="large")

        with col_img:
            st.image(pil_img, caption=infer_img.name, width='stretch')
            h, w = img_array.shape[:2]
            st.caption(f"Resolución: {w}  {h} px")

        with col_res:
            if st.button("✅ Analizar imagen", type="primary", width='stretch'):
                with st.spinner("Ejecutando análisis Hood..."):
                    try:
                        sys.path.insert(0, str(ROOT))
                        from api.inference import HoodInferenceEngine
                        engine = HoodInferenceEngine(str(ONNX_MODEL))
                        result = engine.analyze(img_array)
                        st.session_state["last_infer_result"] = result
                        st.session_state["last_infer_name"]   = infer_img.name
                    except Exception as ex:
                        st.error(f"Error durante la inferencia: {ex}")

            # Mostrar resultado si existe en sesión
            if "last_infer_result" in st.session_state:
                res       = st.session_state["last_infer_result"]
                img_label = st.session_state.get("last_infer_name", "imagen")

                total = int(res.get("total_hood_score", 0))
                pct   = total / 240 * 100

                col_m1, col_m2, col_m3 = st.columns(3)
                col_m1.metric("Score Hood Total", f"{total} / 240")
                col_m2.metric("Porcentaje de daño", f"{pct:.1f}%")
                nivel = "Severo" if pct > 50 else "Moderado" if pct > 20 else "Leve" if pct > 5 else "Mínimo"
                col_m3.metric("Nivel global", nivel)

                st.divider()
                st.subheader("Tabla Hood completa")

                zones_data = res.get("zones", {})
                matrix_rows = {}
                for zone_name in ZONE_NAMES:
                    dmg_dict = zones_data.get(zone_name, {d: 0 for d in DAMAGE_TYPES})
                    matrix_rows[zone_name.replace("_", " ")] = {
                        d: dmg_dict.get(d, 0) for d in DAMAGE_TYPES
                    }

                df_hood = pd.DataFrame(matrix_rows).T
                df_hood.columns = DAMAGE_TYPES

                # Colorear celdas según score (0=gris, 1=amarillo, 2=naranja, 3=rojo)
                def _color_cell(val):
                    c = SCORE_COLORS.get(int(val), "#f5f5f5")
                    return f"background-color:{c}; color:black; text-align:center;"

                styled = df_hood.style.applymap(_color_cell)
                st.dataframe(styled, width='stretch', height=400)

                if _PLOTLY:
                    st.subheader("Mapa de calor")
                    matrix_vals = df_hood.values.tolist()
                    fig_hm = px.imshow(
                        matrix_vals,
                        x=DAMAGE_TYPES,
                        y=list(df_hood.index),
                        color_continuous_scale=[
                            "#f5f5f5", "#fff176", "#ffb74d", "#ef5350"
                        ],
                        zmin=0, zmax=3,
                        text_auto=True,
                        title=f"Scores Hood  {img_label}  (total: {total}/240)",
                        aspect="auto",
                    )
                    fig_hm.update_coloraxes(showscale=True)
                    st.plotly_chart(fig_hm, width='stretch')

                # Botón para descargar el JSON de resultados
                result_json = json.dumps(res, indent=2, ensure_ascii=False)
                st.download_button(
                    "️ Descargar resultado JSON",
                    data=result_json,
                    file_name=f"hood_result_{Path(img_label).stem}.json",
                    mime="application/json",
                    width='stretch',
                )

                st.divider()
                st.subheader("Localización del daño en imagen")

                from api.inference import draw_damage_overlay
                overlay_img = draw_damage_overlay(
                    img_array,
                    res,
                    landmarks=res.get("landmarks"),
                    min_score=1,
                    alpha=0.30,
                )
                # Leyenda de colores
                st.caption(
                    "Color = tipo de daño:  "
                    "Rojo=rayado  Naranja=picado  Amarillo=bruñido  "
                    "Verde=abrasión  Azul=delaminación  Violeta=deformación  Cyan=residuos  |  "
                    "Grosor del borde = severidad: fino=leve · medio=moderado · grueso=severo  |  "
                    "Puntos cyan = landmarks"
                )
                st.image(overlay_img, caption="Zonas con daño detectado", width='stretch')
