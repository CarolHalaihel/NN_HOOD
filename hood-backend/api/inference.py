"""
inference.py — Motor de inferencia Hood NN

Ejecuta el pipeline completo:
  imagen RGB → detección de landmarks → cálculo de zonas → scores Hood → tabla completa

Detección de landmarks:
  Se usa detección clásica por visión computacional (CLAHE + Otsu + minAreaRect)
  como método principal, ya que las imágenes son de setup fotográfico controlado.
  No requiere el LandmarkDetector NN (ahorra tiempo de entrenamiento adicional).

Clasificación de daño:
  Usa el modelo HoodNet exportado en formato ONNX para máxima velocidad en CPU.
"""

import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import onnxruntime as ort
    _ORT_AVAILABLE = True
except ImportError:
    _ORT_AVAILABLE = False

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from api.zone_computer import HoodZoneComputer, ZONE_NAMES, LANDMARK_NAMES

DAMAGE_TYPES = [
    "rayado",       "picado",      "brunido",     "abrasion",
    "delaminacion", "deformacion", "residuos",
]

# Parámetros de normalización ImageNet (deben coincidir con dataset.py)
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESADO
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_crop(image_rgb: np.ndarray, size: int = 224) -> np.ndarray:
    """
    Preprocesa un recorte de zona para entrada al modelo ONNX.

    Entrada:  np.ndarray (H, W, 3) uint8 RGB
    Salida:   np.ndarray (1, 3, size, size) float32 normalizado ImageNet
    """
    img = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    img = img.transpose(2, 0, 1)          # HWC → CHW
    img = np.expand_dims(img, axis=0)     # → (1, 3, H, W)
    return np.ascontiguousarray(img)


# ─────────────────────────────────────────────────────────────────────────────
# DETECCIÓN CLÁSICA DE LANDMARKS
# ─────────────────────────────────────────────────────────────────────────────

def detect_landmarks_classical(image_rgb: np.ndarray) -> dict:
    """
    Detecta los 7 landmarks anatómicos usando visión computacional clásica.
    Optimizado para imágenes con fondo oscuro (fotografía sobre fondo negro/neutro).

    Algoritmo:
      1. CLAHE + Otsu inverso para fondo oscuro → máscara del implante
      2. Contorno más grande → TL, TR, BL, BR via minAreaRect
      3. Erosión morfológica para separar cóndilos → 2 blobs → MC, LC
      4. IG en el punto medio entre los dos cóndilos a la altura del surco

    Retorna: dict con claves "TL", "TR", "BL", "BR", "MC", "LC", "IG"
    """
    h, w = image_rgb.shape[:2]
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

    # ── 1. Segmentar implante del fondo oscuro ────────────────────────────
    clahe    = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Otsu estándar: implante (claro) → blanco, fondo (oscuro) → negro
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Si la imagen tiene fondo blanco, invertir (>50% pixels blancos = fondo blanco)
    if np.mean(binary) > 128:
        binary = cv2.bitwise_not(binary)

    # Cerrar huecos (tornillos, reflejos internos, etc.)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (w // 12, h // 12))
    binary  = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k_close)
    # Rellenar huecos internos completamente
    binary_fill = binary.copy()
    cv2.floodFill(binary_fill, None, (0, 0), 255)
    binary = cv2.bitwise_or(binary, cv2.bitwise_not(binary_fill))

    # ── 2. Contorno más grande → bounding box del implante ───────────────
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _fallback_landmarks(w, h)

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < w * h * 0.04:
        return _fallback_landmarks(w, h)

    rect    = cv2.minAreaRect(largest)
    box_pts = cv2.boxPoints(rect)
    box_pts = _sort_box_points(box_pts.astype(float))  # [TL, TR, BR, BL]
    tl, tr, br, bl = box_pts
    ax, ay, aw, ah = cv2.boundingRect(largest)

    # ── 3. Detectar los 2 cóndilos por blob analysis ─────────────────────
    # Erosión agresiva para separar los cóndilos del surco central
    k_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                         (max(1, aw // 7), max(1, ah // 5)))
    eroded  = cv2.erode(binary[ay:ay+ah, ax:ax+aw], k_erode, iterations=2)

    blob_contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)

    # Trasladar contornos al sistema de coordenadas global
    blob_data = []
    for cnt in blob_contours:
        area = cv2.contourArea(cnt)
        if area < (aw * ah) * 0.03:  # ignorar blobs pequeños
            continue
        M   = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx_ = ax + int(M["m10"] / M["m00"])
        cy_ = ay + int(M["m01"] / M["m00"])
        bx, by, bw_, bh_ = cv2.boundingRect(cnt)
        blob_data.append((area, cx_, cy_, ax + bx, ay + by, bw_, bh_))

    blob_data.sort(key=lambda b: -b[0])  # mayor área primero
    blobs = blob_data[:2]

    if len(blobs) >= 2:
        blobs.sort(key=lambda b: b[1])  # izquierda → derecha
        _, cx1, cy1, bx1, by1, bw1, bh1 = blobs[0]
        _, cx2, cy2, bx2, by2, bw2, bh2 = blobs[1]
        mc = [float(cx1), float(cy1)]
        lc = [float(cx2), float(cy2)]
        # IG: punto medio horizontal entre los dos cóndilos, a la altura del surco
        ig_x = (cx1 + cx2) / 2.0
        # Altura del surco: mínimo de intensidad en la franja central entre los dos cóndilos
        roi_y1 = max(0, min(cy1, cy2) - ah // 6)
        roi_y2 = min(h,  max(cy1, cy2) + ah // 6)
        roi_x1 = max(0, min(cx1, cx2))
        roi_x2 = min(w, max(cx1, cx2))
        if roi_x2 > roi_x1 and roi_y2 > roi_y1:
            roi_gray = enhanced[roi_y1:roi_y2, roi_x1:roi_x2]
            col_mean = roi_gray.mean(axis=1)  # perfil vertical
            ig_y     = roi_y1 + float(np.argmin(col_mean))
        else:
            ig_y = float((cy1 + cy2) / 2)
        ig = [ig_x, ig_y]
    else:
        # Fallback: perfil horizontal si blob detection falla
        mc, lc, ig = _find_condyle_landmarks(enhanced, ax, ay, aw, ah)
        mc = [float(mc[0]), float(mc[1])]
        lc = [float(lc[0]), float(lc[1])]
        ig = [float(ig[0]), float(ig[1])]

    return {
        "TL": [float(tl[0]), float(tl[1])],
        "TR": [float(tr[0]), float(tr[1])],
        "BL": [float(bl[0]), float(bl[1])],
        "BR": [float(br[0]), float(br[1])],
        "MC": mc,
        "LC": lc,
        "IG": ig,
    }


def _sort_box_points(pts: np.ndarray) -> np.ndarray:
    """Ordena 4 puntos como [TL, TR, BR, BL]."""
    rect = np.zeros((4, 2), dtype=float)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    rect[0] = pts[np.argmin(s)]     # TL: menor x+y
    rect[2] = pts[np.argmax(s)]     # BR: mayor x+y
    rect[1] = pts[np.argmin(diff)]  # TR: menor x-y
    rect[3] = pts[np.argmax(diff)]  # BL: mayor x-y
    return rect


def _find_condyle_landmarks(
    gray: np.ndarray,
    x: int, y: int, w: int, h: int,
) -> tuple:
    """
    Localiza los cóndilos (MC, LC) y el surco (IG) mediante análisis de perfil.

    Analiza la franja del tercio superior del implante donde se encuentran los cóndilos.
    Detecta dos máximos de intensidad (cimas de cóndilos) y el valle entre ellos (surco).

    Retorna: ([mc_x, mc_y], [lc_x, lc_y], [ig_x, ig_y]) en píxeles
    """
    # Franja del tercio superior-central del implante (donde están los cóndilos)
    y_top = y + h // 6
    y_bot = y + h // 2
    roi   = gray[y_top:y_bot, x:x + w]

    if roi.size == 0:
        cx = x + w // 2
        cy = y + h // 3
        return [cx - w // 4, cy], [cx + w // 4, cy], [cx, cy]

    # Perfil horizontal: promedio de intensidad por columna
    profile = roi.mean(axis=0).astype(float)

    # Suavizado gaussiano para eliminar ruido
    kernel_size = max(1, w // 15)
    if kernel_size % 2 == 0:
        kernel_size += 1
    smoothed = cv2.GaussianBlur(profile.reshape(1, -1), (kernel_size, 1), 0).flatten()

    cy_condyle = (y_top + y_bot) // 2

    # Buscar picos (cimas de cóndilos) con scipy
    try:
        from scipy.signal import find_peaks
        min_distance = max(1, w // 5)
        peaks, props = find_peaks(smoothed, distance=min_distance, prominence=3.0)
    except ImportError:
        peaks = np.array([])

    if len(peaks) >= 2:
        # Dos o más picos: tomar los dos más prominentes
        if "prominences" in (props if "props" in dir() else {}):
            top2 = sorted(peaks, key=lambda p: -props["prominences"][list(peaks).index(p)])[:2]
        else:
            top2 = peaks[:2]
        p_sorted = sorted(top2)
        mc_x = x + p_sorted[0]
        lc_x = x + p_sorted[1]
        # Surco: mínimo entre los dos picos
        valley_seg = smoothed[p_sorted[0]:p_sorted[1] + 1]
        ig_local   = p_sorted[0] + int(np.argmin(valley_seg))
        ig_x       = x + ig_local
    elif len(peaks) == 1:
        # Solo un pico detectado: dividir simétricamente
        mc_x = x + peaks[0]
        ig_x = x + peaks[0] + w // 4
        lc_x = x + peaks[0] + w // 2
    else:
        # Fallback: dividir en tercios
        mc_x = x + w // 3
        ig_x = x + w // 2
        lc_x = x + 2 * w // 3

    return [mc_x, cy_condyle], [lc_x, cy_condyle], [ig_x, cy_condyle]


def _fallback_landmarks(w: int, h: int) -> dict:
    """Landmarks de emergencia cuando la detección falla completamente."""
    return {
        "TL": [w * 0.10, h * 0.10],
        "TR": [w * 0.90, h * 0.10],
        "BL": [w * 0.10, h * 0.90],
        "BR": [w * 0.90, h * 0.90],
        "MC": [w * 0.30, h * 0.30],
        "LC": [w * 0.70, h * 0.30],
        "IG": [w * 0.50, h * 0.30],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE INFERENCIA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class HoodInferenceEngine:
    """
    Motor de inferencia completo del sistema Hood NN.

    Pipeline: imagen → landmarks → zonas → HoodNet ONNX → tabla Hood completa

    Uso:
      engine = HoodInferenceEngine("models/hood_model.onnx")
      result = engine.analyze(image_rgb_array)
      # result["total_hood_score"] → int (0-240)
      # result["zones"]["medial_anterior"]["abrasion"] → int (0-3)
    """

    def __init__(
        self,
        onnx_path: str,
        use_classical_landmarks: bool = True,
    ):
        """
        onnx_path               : ruta al archivo hood_model.onnx
        use_classical_landmarks : True → detección clásica (recomendado para
                                  setup controlado). False → requiere landmarks
                                  manuales vía analyze(landmarks=...).
        """
        if not _ORT_AVAILABLE:
            raise ImportError(
                "onnxruntime no está instalado. "
                "Instálalo con: pip install onnxruntime"
            )

        onnx_file = Path(onnx_path)
        if not onnx_file.exists():
            raise FileNotFoundError(
                f"Modelo ONNX no encontrado: {onnx_path}\n"
                "Ejecuta primero: python train/export_onnx.py"
            )

        self.session = ort.InferenceSession(
            str(onnx_file),
            providers=["CPUExecutionProvider"],
        )
        self.use_classical_landmarks = use_classical_landmarks
        print(f"[INFO] HoodInferenceEngine listo. Modelo: {onnx_path}")

    def analyze(
        self,
        image_rgb: np.ndarray,
        landmarks: Optional[dict] = None,
    ) -> dict:
        """
        Ejecuta el pipeline completo de análisis Hood sobre una imagen.

        Parámetros:
          image_rgb : np.ndarray (H, W, 3) uint8 en formato RGB
          landmarks : dict opcional con los 7 landmarks en píxeles.
                      Si se omite, se detectan automáticamente (detección clásica).

        Retorna:
          {
            "zones": {
              "medial_anterior": {"delaminacion": 0, "abrasion": 1, ...},
              ...                                  (10 zonas × 7 tipos = 70 valores)
            },
            "total_hood_score": 42,            # suma total (0-240)
            "landmarks": {"TL": [x,y], ...},   # landmarks usados
            "zone_scores_matrix": [[0,1,...], ...]  # lista 10 × 7 de ints
          }
        """
        # ── 1. Detectar / recibir landmarks ──────────────────────────────────
        if landmarks is not None:
            lm = landmarks
        elif self.use_classical_landmarks:
            lm = detect_landmarks_classical(image_rgb)
        else:
            raise ValueError(
                "No se proporcionaron landmarks y use_classical_landmarks=False. "
                "Pasa landmarks=dict(...) o usa use_classical_landmarks=True."
            )

        # ── 2. Calcular geometría de zonas ────────────────────────────────────
        zone_computer = HoodZoneComputer(lm)

        # ── 3. Por cada zona: recortar → preprocesar → inferir ONNX ──────────
        zone_scores    = {}
        scores_matrix  = []

        for zone_idx, zone_name in enumerate(ZONE_NAMES):
            crop         = zone_computer.get_zone_crop(image_rgb, zone_idx, output_size=224)
            input_tensor = preprocess_crop(crop, size=224)

            outputs = self.session.run(None, {"image": input_tensor})
            logits  = outputs[0]  # (1, 7, 4)

            # Score = argmax sobre la dimensión de clases (0, 1, 2, 3)
            scores = logits[0].argmax(axis=-1).tolist()  # lista de 7 ints

            zone_scores[zone_name] = {
                dmg: int(score)
                for dmg, score in zip(DAMAGE_TYPES, scores)
            }
            scores_matrix.append(scores)

        # ── 4. Score total Hood ───────────────────────────────────────────────
        total_score = sum(
            score
            for zone in zone_scores.values()
            for score in zone.values()
        )

        return {
            "zones":              zone_scores,
            "total_hood_score":   total_score,
            "landmarks":          lm,
            "zone_scores_matrix": scores_matrix,
        }


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACIÓN — OVERLAY DE DAÑO SOBRE LA IMAGEN ORIGINAL
# ─────────────────────────────────────────────────────────────────────────────

# Color RGB por tipo de daño (7 colores distintos)
_DAMAGE_COLOR_RGB = {
    "rayado":       (255,  80,  80),   # rojo
    "picado":       (255, 165,   0),   # naranja
    "brunido":      (255, 230,   0),   # amarillo
    "abrasion":     ( 60, 200,  60),   # verde
    "delaminacion": ( 60, 160, 255),   # azul claro
    "deformacion":  (180,  60, 255),   # violeta
    "residuos":     ( 60, 230, 200),   # cyan
}

# Grosor del borde según severidad
_SEV_THICKNESS = {1: 2, 2: 4, 3: 7}

# Nombres cortos para etiquetas
_DAMAGE_SHORT = {
    "rayado": "ray", "picado": "pic", "brunido": "bru", "abrasion": "abr",
    "delaminacion": "del", "deformacion": "def", "residuos": "res",
}
_SEVERITY_LABEL = {1: "1-leve", 2: "2-mod", 3: "3-sev"}


def draw_damage_overlay(
    image_rgb: np.ndarray,
    result: dict,
    landmarks: Optional[dict] = None,
    min_score: int = 1,
    alpha: float = 0.15,
) -> np.ndarray:
    """
    Dibuja un rectángulo por cada (zona × tipo_de_daño) activo.

    - Color del rectángulo → tipo de daño (7 colores fijos, ver _DAMAGE_COLOR_RGB)
    - Grosor del borde     → severidad: fino=leve(1), medio=mod(2), grueso=sev(3)
    - Relleno semitransparente del mismo color
    - Si hay varios daños en la misma zona, cada rect se retranquea 5 px
      hacia el interior para que todos sean visibles
    - Etiqueta en esquina superior izquierda: "ray 2-mod" (color del daño)
    - Landmarks como puntos cyan con nombre
    """
    img     = image_rgb.copy()
    overlay = img.copy()

    lm = landmarks or result.get("landmarks", {})
    if not lm:
        return img
    try:
        zc = HoodZoneComputer(lm)
    except Exception:
        return img

    zones_data = result.get("zones", {})
    font       = cv2.FONT_HERSHEY_SIMPLEX
    fs         = max(0.30, min(image_rgb.shape[:2]) / 1600)
    lbl_th     = max(1, int(fs * 2))

    for zone_idx, zone_name in enumerate(ZONE_NAMES):
        scores = zones_data.get(zone_name, {})
        # Daños activos ordenados de mayor a menor severidad
        active = sorted(
            [(dmg, sc) for dmg, sc in scores.items() if sc >= min_score],
            key=lambda x: -x[1],
        )
        if not active:
            continue

        x1b, y1b, x2b, y2b = zc.get_zone_bbox(zone_idx)
        x1b = max(0, x1b); y1b = max(0, y1b)
        x2b = min(img.shape[1] - 1, x2b); y2b = min(img.shape[0] - 1, y2b)
        if x2b <= x1b or y2b <= y1b:
            continue

        for draw_i, (dmg, sev) in enumerate(active):
            # Retranqueo: cada daño adicional 5 px hacia el interior
            inset = draw_i * 5
            x1 = x1b + inset; y1 = y1b + inset
            x2 = x2b - inset; y2 = y2b - inset
            if x2 <= x1 + 4 or y2 <= y1 + 4:
                continue

            r, g, b   = _DAMAGE_COLOR_RGB.get(dmg, (200, 200, 200))
            color_bgr = (b, g, r)                      # OpenCV usa BGR
            bord_th   = _SEV_THICKNESS.get(sev, 2)

            # Relleno semitransparente sobre overlay
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color_bgr, -1)
            # Borde sólido cuyo grosor indica la severidad
            cv2.rectangle(img,     (x1, y1), (x2, y2), color_bgr, bord_th)

            # Etiqueta interior: "ray 2-mod"
            label = f"{_DAMAGE_SHORT.get(dmg, dmg)} {_SEVERITY_LABEL.get(sev, str(sev))}"
            (tw, th), _ = cv2.getTextSize(label, font, fs, lbl_th)
            ty = y1 + th + 3
            # Fondo negro detrás del texto
            cv2.rectangle(img, (x1 + 2, y1 + 1), (x1 + tw + 6, y1 + th + 6),
                          (0, 0, 0), -1)
            cv2.putText(img, label, (x1 + 4, ty), font, fs,
                        (r, g, b), lbl_th, cv2.LINE_AA)

    # Mezclar relleno semitransparente
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    # Landmarks como puntos cyan con nombre
    if lm:
        for name, coords in lm.items():
            cx, cy = int(coords[0]), int(coords[1])
            cv2.circle(img, (cx, cy), max(5, lbl_th * 3), (0, 210, 255), -1)
            cv2.putText(img, name, (cx + 6, cy - 4), font, fs * 0.85,
                        (0, 210, 255), max(1, lbl_th - 1), cv2.LINE_AA)

    return img
