"""
four_zone_computer.py — FourZoneComputer

Calcula 4 zonas de la bandeja tibial a partir de 4 landmarks:
  AM (AnteriorMedial) - Superior Izquierda
  AL (AnteriorLateral) - Superior Derecha
  PM (PosteriorMedial) - Inferior Izquierda
  PL (PosteriorLateral) - Inferior Derecha

Útil para segmentación de la vista inferior (_bE).
"""

from typing import Optional, Tuple

import cv2
import numpy as np

FOUR_ZONE_NAMES = ["AM", "AL", "PM", "PL"]
FOUR_LANDMARK_NAMES = ["AM", "AL", "PM", "PL"]

NUM_FOUR_ZONES = 4
ZONE_CROP_SIZE = 224


class FourZoneComputer:
    """
    Calcula 4 zonas a partir de 4 landmarks.

    landmarks: dict con claves "AM", "AL", "PM", "PL"
               y valores [x, y] en píxeles de la imagen original.

    API:
      bbox = zc.get_zone_bbox(zone_idx)          # (x1, y1, x2, y2)
      crop = zc.get_zone_crop(image_rgb, zone_idx)  # np.ndarray (224, 224, 3)
      vis  = zc.draw_zones(image_rgb)            # imagen anotada
    """

    def __init__(self, landmarks: dict, image_rgb: Optional[np.ndarray] = None):
        """
        landmarks: {"AM": [x,y], "AL": [x,y], "PM": [x,y], "PL": [x,y]}
        image_rgb: imagen para detectar automáticamente los límites de la prótesis
        """
        self._parse_landmarks(landmarks)
        self.image_rgb = image_rgb
        self._compute_geometry()

    def _parse_landmarks(self, landmarks: dict):
        """Valida y extrae los 4 landmarks."""
        for key in FOUR_LANDMARK_NAMES:
            if key not in landmarks:
                raise ValueError(f"Landmark '{key}' ausente. Requeridos: {FOUR_LANDMARK_NAMES}")
        
        self.AM = np.array(landmarks["AM"], dtype=float)  # Superior izquierda
        self.AL = np.array(landmarks["AL"], dtype=float)  # Superior derecha
        self.PM = np.array(landmarks["PM"], dtype=float)  # Inferior izquierda
        self.PL = np.array(landmarks["PL"], dtype=float)  # Inferior derecha

    def _compute_geometry(self):
        """Calcula los límites y centros de las 4 zonas."""
        # Si tenemos la imagen, detectar automáticamente los límites de la prótesis
        if self.image_rgb is not None:
            self._detect_prosthesis_bounds()
        else:
            # Fallback: usar los landmarks como límites
            self.x_min = min(self.AM[0], self.AL[0], self.PM[0], self.PL[0])
            self.x_max = max(self.AM[0], self.AL[0], self.PM[0], self.PL[0])
            self.y_min = min(self.AM[1], self.AL[1], self.PM[1], self.PL[1])
            self.y_max = max(self.AM[1], self.AL[1], self.PM[1], self.PL[1])

        # Punto central como divisor (media de los 4 landmarks)
        self.center_x = (self.AM[0] + self.AL[0] + self.PM[0] + self.PL[0]) / 4.0
        self.center_y = (self.AM[1] + self.AL[1] + self.PM[1] + self.PL[1]) / 4.0

    def _detect_prosthesis_bounds(self):
        """
        Detecta automáticamente los límites de la prótesis (área no-negra).
        """
        if self.image_rgb is None:
            return
        
        # Convertir a escala de grises
        gray = cv2.cvtColor(self.image_rgb, cv2.COLOR_RGB2GRAY)
        
        # Crear máscara de área no-negra (no es completamente negra)
        # Umbral bajo para detectar la prótesis (gris/blanco)
        threshold = 30
        mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Aplicar morfología para limpiar ruido
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Obtener el contorno más grande (la prótesis)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Establecer límites con pequeño margen
            margin = 5
            self.x_min = max(0, x - margin)
            self.y_min = max(0, y - margin)
            self.x_max = min(self.image_rgb.shape[1], x + w + margin)
            self.y_max = min(self.image_rgb.shape[0], y + h + margin)
        else:
            # Si no detecta, usar landmarks
            self.x_min = min(self.AM[0], self.AL[0], self.PM[0], self.PL[0])
            self.x_max = max(self.AM[0], self.AL[0], self.PM[0], self.PL[0])
            self.y_min = min(self.AM[1], self.AL[1], self.PM[1], self.PL[1])
            self.y_max = max(self.AM[1], self.AL[1], self.PM[1], self.PL[1])

    def get_zone_bbox(self, zone_idx: int, margin: float = 10.0) -> Tuple[int, int, int, int]:
        """
        Retorna el bounding box (x1, y1, x2, y2) para una zona.
        
        zone_idx: 0=AM, 1=AL, 2=PM, 3=PL
        """
        if zone_idx == 0:  # AM: Superior-Izquierda
            x1 = max(0, int(self.x_min - margin))
            y1 = max(0, int(self.y_min - margin))
            x2 = int(self.center_x + margin)
            y2 = int(self.center_y + margin)
        elif zone_idx == 1:  # AL: Superior-Derecha
            x1 = int(self.center_x - margin)
            y1 = max(0, int(self.y_min - margin))
            x2 = int(self.x_max + margin)
            y2 = int(self.center_y + margin)
        elif zone_idx == 2:  # PM: Inferior-Izquierda
            x1 = max(0, int(self.x_min - margin))
            y1 = int(self.center_y - margin)
            x2 = int(self.center_x + margin)
            y2 = int(self.y_max + margin)
        elif zone_idx == 3:  # PL: Inferior-Derecha
            x1 = int(self.center_x - margin)
            y1 = int(self.center_y - margin)
            x2 = int(self.x_max + margin)
            y2 = int(self.y_max + margin)
        else:
            raise ValueError(f"Índice de zona inválido: {zone_idx}. Use 0-3.")

        return (x1, y1, x2, y2)

    def get_zone_crop(self, image_rgb: np.ndarray, zone_idx: int, output_size: int = 224) -> np.ndarray:
        """
        Extrae un recorte cuadrado centrado en una zona y lo redimensiona.
        
        image_rgb: np.ndarray (H, W, 3) uint8 RGB
        zone_idx: 0=AM, 1=AL, 2=PM, 3=PL
        output_size: tamaño de salida (default 224)
        """
        x1, y1, x2, y2 = self.get_zone_bbox(zone_idx, margin=5)
        
        # Clamp a límites de imagen
        H, W = image_rgb.shape[:2]
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(x1 + 1, min(x2, W))
        y2 = max(y1 + 1, min(y2, H))

        crop = image_rgb[y1:y2, x1:x2, :]
        if crop.size == 0:
            return np.zeros((output_size, output_size, 3), dtype=np.uint8)

        crop_resized = cv2.resize(crop, (output_size, output_size))
        return crop_resized

    def draw_zones(self, image_rgb: np.ndarray, radius: int = 5, thickness: int = 2) -> np.ndarray:
        """
        Dibuja los 4 cuadrantes (AM, AL, PM, PL) con líneas de división y etiquetas.
        """
        vis = image_rgb.copy()
        
        # Colores para cada zona (BGR)
        zone_colors = {
            0: (200, 100, 100),    # AM: Azul (BGR)
            1: (100, 200, 100),    # AL: Verde
            2: (100, 100, 200),    # PM: Rojo
            3: (200, 200, 100),    # PL: Cian
        }
        
        zone_names = ["AM", "AL", "PM", "PL"]

        # Centro de división
        center_x_int = int(self.center_x)
        center_y_int = int(self.center_y)
        x_min_int = int(self.x_min)
        x_max_int = int(self.x_max)
        y_min_int = int(self.y_min)
        y_max_int = int(self.y_max)

        # Dibujar rectángulos sombreados para cada zona
        # AM (0): arriba-izquierda
        cv2.rectangle(vis, (x_min_int, y_min_int), (center_x_int, center_y_int),
                     zone_colors[0], -1)
        # AL (1): arriba-derecha
        cv2.rectangle(vis, (center_x_int, y_min_int), (x_max_int, center_y_int),
                     zone_colors[1], -1)
        # PM (2): abajo-izquierda
        cv2.rectangle(vis, (x_min_int, center_y_int), (center_x_int, y_max_int),
                     zone_colors[2], -1)
        # PL (3): abajo-derecha
        cv2.rectangle(vis, (center_x_int, center_y_int), (x_max_int, y_max_int),
                     zone_colors[3], -1)

        # Mezclar con la imagen original (50% transparencia)
        vis = cv2.addWeighted(vis, 0.4, image_rgb, 0.6, 0)

        # Líneas de división gruesas
        line_color = (255, 255, 255)  # Blanco
        line_thickness = 3
        
        # Línea vertical
        cv2.line(vis, (center_x_int, y_min_int), (center_x_int, y_max_int),
                line_color, line_thickness)
        # Línea horizontal
        cv2.line(vis, (x_min_int, center_y_int), (x_max_int, center_y_int),
                line_color, line_thickness)

        # Etiquetas de zonas en el centro de cada cuadrante
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        font_thickness = 2
        font_color = (255, 255, 255)  # Blanco

        # AM (arriba-izquierda)
        text_x = int((x_min_int + center_x_int) / 2)
        text_y = int((y_min_int + center_y_int) / 2)
        text_size = cv2.getTextSize("AM", font, font_scale, font_thickness)[0]
        cv2.putText(vis, "AM", (text_x - text_size[0]//2, text_y + text_size[1]//2),
                   font, font_scale, font_color, font_thickness)

        # AL (arriba-derecha)
        text_x = int((center_x_int + x_max_int) / 2)
        text_y = int((y_min_int + center_y_int) / 2)
        text_size = cv2.getTextSize("AL", font, font_scale, font_thickness)[0]
        cv2.putText(vis, "AL", (text_x - text_size[0]//2, text_y + text_size[1]//2),
                   font, font_scale, font_color, font_thickness)

        # PM (abajo-izquierda)
        text_x = int((x_min_int + center_x_int) / 2)
        text_y = int((center_y_int + y_max_int) / 2)
        text_size = cv2.getTextSize("PM", font, font_scale, font_thickness)[0]
        cv2.putText(vis, "PM", (text_x - text_size[0]//2, text_y + text_size[1]//2),
                   font, font_scale, font_color, font_thickness)

        # PL (abajo-derecha)
        text_x = int((center_x_int + x_max_int) / 2)
        text_y = int((center_y_int + y_max_int) / 2)
        text_size = cv2.getTextSize("PL", font, font_scale, font_thickness)[0]
        cv2.putText(vis, "PL", (text_x - text_size[0]//2, text_y + text_size[1]//2),
                   font, font_scale, font_color, font_thickness)

        # Dibujar los 4 puntos como referencia
        landmark_radius = 8
        landmark_thickness = 2
        for i, (name, pt) in enumerate([("AM", self.AM), ("AL", self.AL), 
                                         ("PM", self.PM), ("PL", self.PL)]):
            pt_int = tuple(map(int, pt))
            cv2.circle(vis, pt_int, landmark_radius, (255, 255, 255), -1)
            cv2.circle(vis, pt_int, landmark_radius, (0, 0, 0), landmark_thickness)

        return vis
