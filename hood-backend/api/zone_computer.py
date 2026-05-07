"""
zone_computer.py — HoodZoneComputer

Calcula las 10 zonas anatómicas del método Hood a partir de 7 landmarks.
Geometría basada en sectores diagonales de elipse (patrón X) sobre cada cóndilo,
siguiendo el diagrama de referencia Hood.

Numeración Hood (índice interno = número Hood):
  Cóndilo medial:   0=periférico  1=anterior  2=central  3=posterior
  Cóndilo lateral:  4=central     5=anterior  6=periférico  7=posterior
  Surcos:           8=anterior    9=posterior

Dentro de cada cóndilo la elipse se divide con dos diagonales a 45° (patrón X):
  - Sector norte  (|ny| > |nx|, ny < 0)  → Anterior
  - Sector sur    (|ny| > |nx|, ny > 0)  → Posterior
  - Sector inner  (|nx| > |ny|, hacia surco)  → Central
  - Sector outer  (|nx| > |ny|, alejado surco) → Periférico

  donde nx = (px-cx)/rx,  ny = (py-cy)/ry  (normalizado a la elipse)

Convención de imagen:
  - Eje Y crece hacia abajo (coordenadas de píxel estándar)
  - "Anterior" = parte superior de la imagen
  - "Posterior" = parte inferior
"""

from typing import Optional

import cv2
import numpy as np

ZONE_NAMES = [
    "medial_periferico",  "medial_anterior",   "medial_central",    "medial_posterior",
    "lateral_central",    "lateral_anterior",  "lateral_periferico","lateral_posterior",
    "surco_anterior",     "surco_posterior",
]

LANDMARK_NAMES = ["TL", "TR", "BL", "BR", "MC", "LC", "IG"]

NUM_ZONES      = 10
ZONE_CROP_SIZE = 224


class HoodZoneComputer:
    """
    Calcula las 10 zonas Hood a partir de 7 landmarks anatómicos.

    landmarks: dict con claves "TL","TR","BL","BR","MC","LC","IG"
               y valores [x, y] en píxeles de la imagen original.

    API:
      bbox = zc.get_zone_bbox(zone_idx)          # (x1,y1,x2,y2)
      crop = zc.get_zone_crop(image_rgb, 3)       # np.ndarray (224,224,3)
      vis  = zc.draw_zones(image_rgb)             # imagen anotada
    """

    # Semi-ejes del cóndilo como fracción del tamaño del implante.
    # El cóndilo tibial está elongado en dirección AP (vertical en la imagen),
    # por lo que RY_FRAC >> RX_FRAC para cualquier aspecto de foto.
    # Ejemplo 1200×600 px: rx=156 px, ry=252 px → ratio 1.6× más alto que ancho.
    CONDYLE_RX_FRAC = 0.13
    CONDYLE_RY_FRAC = 0.42

    def __init__(self, landmarks: dict):
        self._parse_landmarks(landmarks)
        self._compute_geometry()

    def _parse_landmarks(self, landmarks: dict):
        for key in LANDMARK_NAMES:
            if key not in landmarks:
                raise ValueError(f"Landmark '{key}' ausente. Requeridos: {LANDMARK_NAMES}")
        self.TL = np.array(landmarks["TL"], dtype=float)
        self.TR = np.array(landmarks["TR"], dtype=float)
        self.BL = np.array(landmarks["BL"], dtype=float)
        self.BR = np.array(landmarks["BR"], dtype=float)
        self.MC = np.array(landmarks["MC"], dtype=float)
        self.LC = np.array(landmarks["LC"], dtype=float)
        self.IG = np.array(landmarks["IG"], dtype=float)

    def _compute_geometry(self):
        self.x_min = float(min(self.TL[0], self.BL[0]))
        self.x_max = float(max(self.TR[0], self.BR[0]))
        self.y_min = float(min(self.TL[1], self.TR[1]))
        self.y_max = float(max(self.BL[1], self.BR[1]))
        self.impl_w = self.x_max - self.x_min
        self.impl_h = self.y_max - self.y_min

        self.x_divider      = float(self.IG[0])
        self.medial_is_left = self.MC[0] < self.x_divider

        self.med_cx = float(self.MC[0])
        self.med_cy = float(self.MC[1])
        self.lat_cx = float(self.LC[0])
        self.lat_cy = float(self.LC[1])

        self.med_rx = self.impl_w * self.CONDYLE_RX_FRAC
        self.med_ry = self.impl_h * self.CONDYLE_RY_FRAC
        self.lat_rx = self.impl_w * self.CONDYLE_RX_FRAC
        self.lat_ry = self.impl_h * self.CONDYLE_RY_FRAC

        if self.medial_is_left:
            gx1 = self.med_cx + self.med_rx * 0.55
            gx2 = self.lat_cx - self.lat_rx * 0.55
        else:
            gx1 = self.lat_cx + self.lat_rx * 0.55
            gx2 = self.med_cx - self.med_rx * 0.55
        self.groove_x1    = float(max(self.x_min, min(gx1, gx2)))
        self.groove_x2    = float(min(self.x_max, max(gx1, gx2)))
        self.groove_y_mid = float(self.IG[1])

    # ─────────────────────────────────────────────────────────────────────────
    # GEOMETRÍA DE SECTOR
    # ─────────────────────────────────────────────────────────────────────────

    def _sector_for_zone(self, zone_idx: int) -> tuple:
        if zone_idx in (0, 1, 2, 3):
            return ('medial', {0: 'outer', 1: 'north', 2: 'inner', 3: 'south'}[zone_idx])
        if zone_idx in (4, 5, 6, 7):
            return ('lateral', {4: 'inner', 5: 'north', 6: 'outer', 7: 'south'}[zone_idx])
        return ('groove', 'anterior' if zone_idx == 8 else 'posterior')

    def _resolve_sector(self, side: str, sector: str) -> str:
        if sector in ('north', 'south'):
            return sector
        if side == 'medial':
            inner_dir = 'east' if self.medial_is_left else 'west'
            outer_dir = 'west' if self.medial_is_left else 'east'
        else:
            inner_dir = 'west' if self.medial_is_left else 'east'
            outer_dir = 'east' if self.medial_is_left else 'west'
        return inner_dir if sector == 'inner' else outer_dir

    def _condyle_params(self, side: str) -> tuple:
        if side == 'medial':
            return self.med_cx, self.med_cy, self.med_rx, self.med_ry
        return self.lat_cx, self.lat_cy, self.lat_rx, self.lat_ry

    def _sector_bbox(self, cx, cy, rx, ry, cardinal: str) -> tuple:
        if cardinal == 'north':
            return (int(cx - rx), int(cy - ry), int(cx + rx), int(cy))
        if cardinal == 'south':
            return (int(cx - rx), int(cy),      int(cx + rx), int(cy + ry))
        if cardinal == 'east':
            return (int(cx),      int(cy - ry), int(cx + rx), int(cy + ry))
        return     (int(cx - rx), int(cy - ry), int(cx),      int(cy + ry))

    def _sector_polygon(self, cx, cy, rx, ry, cardinal: str) -> np.ndarray:
        """Triangle formed by two corner-to-corner diagonals of the condyle bounding box."""
        cx, cy, rx, ry = int(cx), int(cy), int(rx), int(ry)
        tl = (cx - rx, cy - ry)
        tr = (cx + rx, cy - ry)
        bl = (cx - rx, cy + ry)
        br = (cx + rx, cy + ry)
        ct = (cx, cy)
        sector_pts = {'north': [tl, tr, ct],
                      'south': [bl, br, ct],
                      'east':  [tr, br, ct],
                      'west':  [tl, bl, ct]}
        return np.array(sector_pts[cardinal], dtype=np.int32).reshape(-1, 1, 2)

    def _groove_polygon(self, part: str) -> np.ndarray:
        x1, y1, x2, y2 = self._groove_bbox_raw(part)
        return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]], dtype=np.int32).reshape(-1,1,2)

    def _groove_bbox_raw(self, part: str) -> tuple:
        if part == 'anterior':
            return (int(self.groove_x1), int(self.y_min),
                    int(self.groove_x2), int(self.groove_y_mid))
        return     (int(self.groove_x1), int(self.groove_y_mid),
                    int(self.groove_x2), int(self.y_max))

    # ─────────────────────────────────────────────────────────────────────────
    # API PÚBLICA — BOUNDING BOXES
    # ─────────────────────────────────────────────────────────────────────────

    def get_zone_bbox(self, zone_idx: int) -> tuple:
        if not (0 <= zone_idx < NUM_ZONES):
            raise ValueError(f"zone_idx debe estar entre 0 y 9, recibido: {zone_idx}")
        side, sector = self._sector_for_zone(zone_idx)
        if side == 'groove':
            return self._groove_bbox_raw(sector)
        cx, cy, rx, ry = self._condyle_params(side)
        cardinal = self._resolve_sector(side, sector)
        return self._sector_bbox(cx, cy, rx, ry, cardinal)

    def get_all_bboxes(self) -> list:
        return [self.get_zone_bbox(i) for i in range(NUM_ZONES)]

    # ─────────────────────────────────────────────────────────────────────────
    # API PÚBLICA — RECORTES DE IMAGEN
    # ─────────────────────────────────────────────────────────────────────────

    def get_zone_crop(self, image: np.ndarray, zone_idx: int,
                      output_size: Optional[int] = None) -> np.ndarray:
        size         = output_size  # None → native resolution
        h_img, w_img = image.shape[:2]
        x1, y1, x2, y2 = self.get_zone_bbox(zone_idx)
        x1 = max(0, x1);  y1 = max(0, y1)
        x2 = min(w_img, x2);  y2 = min(h_img, y2)
        if x2 <= x1 or y2 <= y1:
            return np.zeros((size, size, 3), dtype=np.uint8)
        crop = image[y1:y2, x1:x2].copy()
        side, sector = self._sector_for_zone(zone_idx)
        if side != 'groove':
            cx, cy, rx, ry = self._condyle_params(side)
            cardinal = self._resolve_sector(side, sector)
            poly = self._sector_polygon(cx, cy, rx, ry, cardinal)
            mask = np.zeros((h_img, w_img), dtype=np.uint8)
            cv2.fillPoly(mask, [poly], 255)
            crop[mask[y1:y2, x1:x2] == 0] = 0
        if size is None:
            return crop
        return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)

    def get_all_zone_crops(self, image: np.ndarray,
                           output_size: Optional[int] = None) -> list:
        return [self.get_zone_crop(image, i, output_size) for i in range(NUM_ZONES)]

    # ─────────────────────────────────────────────────────────────────────────
    # VISUALIZACIÓN
    # ─────────────────────────────────────────────────────────────────────────

    def draw_zones(self, image: np.ndarray) -> np.ndarray:
        """
        Dibuja las 10 zonas Hood con sectores diagonales (patrón X) y elipses.
        Muestra el número Hood centrado en cada sector y etiquetas MEDIAL/LATERAL.
        """
        vis = image.copy()

        # Colores BGR por zona
        FILL = [
            (200, 120,  60),  # 0 med periférico
            (255, 200, 100),  # 1 med anterior
            (200, 170,  80),  # 2 med central
            (255, 160,  80),  # 3 med posterior
            ( 80, 140, 230),  # 4 lat central
            (100, 210, 255),  # 5 lat anterior
            ( 60, 100, 200),  # 6 lat periférico
            ( 80, 160, 255),  # 7 lat posterior
            (100, 240, 200),  # 8 surco anterior
            ( 60, 200, 160),  # 9 surco posterior
        ]

        # Relleno semitransparente
        overlay = vis.copy()
        for i in range(NUM_ZONES):
            side, sector = self._sector_for_zone(i)
            if side == 'groove':
                poly = self._groove_polygon(sector)
            else:
                cx, cy, rx, ry = self._condyle_params(side)
                poly = self._sector_polygon(cx, cy, rx, ry,
                                            self._resolve_sector(side, sector))
            cv2.fillPoly(overlay, [poly], FILL[i])
        cv2.addWeighted(overlay, 0.35, vis, 0.65, 0, vis)

        # Bordes y números
        offset_map = {'north': (0, -0.65), 'south': (0, 0.65),
                      'east':  (0.65, 0),  'west':  (-0.65, 0)}
        for i in range(NUM_ZONES):
            side, sector = self._sector_for_zone(i)
            if side == 'groove':
                poly = self._groove_polygon(sector)
                cv2.polylines(vis, [poly], True, FILL[i], 2, cv2.LINE_AA)
                cx_z = int((poly[:,0,0].min() + poly[:,0,0].max()) / 2)
                cy_z = int((poly[:,0,1].min() + poly[:,0,1].max()) / 2)
            else:
                cx, cy, rx, ry = self._condyle_params(side)
                cardinal = self._resolve_sector(side, sector)
                poly = self._sector_polygon(cx, cy, rx, ry, cardinal)
                cv2.polylines(vis, [poly], True, FILL[i], 2, cv2.LINE_AA)
                odx, ody = offset_map[cardinal]
                cx_z = int(cx + rx * odx)
                cy_z = int(cy + ry * ody)

            fs = max(0.55, min(1.4, min(self.med_rx, self.med_ry) / 50))
            th = max(1, int(fs * 2))
            num = str(i)
            (tw, tht), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
            tx, ty = cx_z - tw // 2, cy_z + tht // 2
            cv2.putText(vis, num, (tx+1, ty+1), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (0,0,0), th+1, cv2.LINE_AA)
            cv2.putText(vis, num, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                        (255,255,255), th, cv2.LINE_AA)

        # Elipses de cóndilo
        for cx, cy, rx, ry in [
            (self.med_cx, self.med_cy, self.med_rx, self.med_ry),
            (self.lat_cx, self.lat_cy, self.lat_rx, self.lat_ry),
        ]:
            cv2.ellipse(vis, (int(cx), int(cy)), (int(rx), int(ry)),
                        0, 0, 360, (255,255,255), 2, cv2.LINE_AA)

        # Diagonales X de esquina a esquina del bounding box del cóndilo
        for cx, cy, rx, ry in [
            (self.med_cx, self.med_cy, self.med_rx, self.med_ry),
            (self.lat_cx, self.lat_cy, self.lat_rx, self.lat_ry),
        ]:
            tl = (int(cx - rx), int(cy - ry))
            tr = (int(cx + rx), int(cy - ry))
            bl = (int(cx - rx), int(cy + ry))
            br = (int(cx + rx), int(cy + ry))
            cv2.line(vis, tl, br, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.line(vis, tr, bl, (255, 255, 255), 1, cv2.LINE_AA)

        # Línea divisora y etiquetas MEDIAL/LATERAL
        cv2.line(vis, (int(self.x_divider), int(self.y_min)),
                 (int(self.x_divider), int(self.y_max)), (255,255,0), 1, cv2.LINE_AA)
        y_lbl = max(18, int(self.y_min) - 6)
        for text, cx_l in [("MEDIAL", int(self.med_cx)), ("LATERAL", int(self.lat_cx))]:
            (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            tx = cx_l - tw // 2
            cv2.putText(vis, text, (tx+1, y_lbl+1), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, text, (tx, y_lbl), cv2.FONT_HERSHEY_SIMPLEX,
                        0.52, (255,255,0), 1, cv2.LINE_AA)

        # Landmarks
        lm_pts = {"TL": self.TL, "TR": self.TR, "BL": self.BL, "BR": self.BR,
                  "MC": self.MC, "LC": self.LC, "IG": self.IG}
        lm_col = {"TL": (80,80,255), "TR": (80,80,255), "BL": (80,80,255),
                  "BR": (80,80,255), "MC": (80,220,80), "LC": (80,180,255),
                  "IG": (0,210,255)}
        for nm, pt in lm_pts.items():
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(vis, (px, py), 7, (255,255,255), -1)
            cv2.circle(vis, (px, py), 6, lm_col[nm], -1)
            cv2.putText(vis, nm, (px+9, py-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (0,0,0), 2, cv2.LINE_AA)
            cv2.putText(vis, nm, (px+9, py-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (255,255,255), 1, cv2.LINE_AA)

        return vis
