# 📋 Resumen de Cambios: Soporte para Imágenes Arriba/Abajo

## 🎯 Objetivo
Extender el sistema Hood NN para distinguir entre imágenes de la **parte superior (_aE)** y **parte inferior (_bE)** de la prótesis, con segmentación adaptativa:
- **Arriba (_aE)**: 10 zonas Hood estándar + 7 landmarks
- **Abajo (_bE)**: 4 cuadrantes (AM, AL, PM, PL) + 4 puntos

---

## 📂 Archivos Creados

### 1. `api/four_zone_computer.py` ✨ (NUEVO)
- Clase `FourZoneComputer`: Calcula 4 zonas a partir de 4 landmarks
- Métodos:
  - `get_zone_bbox(zone_idx)`: Retorna bounding box de cada zona
  - `get_zone_crop(image_rgb, zone_idx)`: Extrae recorte cuadrado (224×224)
  - `draw_zones(image_rgb)`: Visualiza los 4 puntos y líneas divisoras
- Zonas:
  - `zona_0` = AM (AnteriorMedial) - Superior Izquierda
  - `zona_1` = AL (AnteriorLateral) - Superior Derecha
  - `zona_2` = PM (PosteriorMedial) - Inferior Izquierda
  - `zona_3` = PL (PosteriorLateral) - Inferior Derecha

### 2. `remove_fatiga.py` y `add_example_bottom.py`
- `remove_fatiga.py`: Script para eliminar columna "Fatiga"
- `add_example_bottom.py`: Añade ejemplo de imagen _bE con 4 zonas

---

## 🔧 Cambios en `app.py`

### Funciones Nuevas
```python
def _get_image_type(img_name: str) -> str
    # Detecta si es arriba (_aE) o abajo (_bE) por extensión

def _default_annotation(img_name: str) -> dict
    # Crea estructura dinámicamente según tipo de imagen
    # Arriba: 10 zonas, 7 landmarks
    # Abajo: 4 zonas, 4 landmarks

def four_landmarks_to_zone_centers(landmarks: dict) -> dict
    # Convierte 4 landmarks (AM, AL, PM, PL) a 4 centros de zona
```

### Cambios en la Interfaz

#### 1️⃣ Selección de Landmarks (Dinámico)
- **Arriba**: 2 filas × 5 botones (10 zonas)
- **Abajo**: 2 columnas × 2 botones (4 puntos)

```python
# Detección automática del tipo
img_type = _get_image_type(selected_img)
if img_type == "abajo":
    max_zones = 4
    landmark_labels = {"zona_0": "AM", "zona_1": "AL", ...}
else:
    max_zones = 10
    landmark_labels = ZONE_CLICK_LABELS
```

#### 2️⃣ Visualización de Zonas (Dinámico)
- **Arriba**: Usa `HoodZoneComputer` (10 zonas)
- **Abajo**: Usa `FourZoneComputer` (4 zonas)

```python
if img_type_vis == "abajo":
    zc = FourZoneComputer(zone_centers)
else:
    zc = HoodZoneComputer(...)
```

#### 3️⃣ Tabla de Scores (Dinámico)
- **Arriba**: 10 filas × 7 columnas (alquiler 420px)
- **Abajo**: 4 filas × 7 columnas (altura 180px)

```python
num_zones = 4 if img_type == "abajo" else 10
max_score = 3 * 7 * num_zones  # Cálculo dinámico
```

#### 4️⃣ Recortes de Zonas (Dinámico)
- **Arriba**: 2 expansores con 2×5 recortes
- **Abajo**: 1 expansor con 2×2 recortes

---

## 📊 Estructura de Anotaciones

### Imagen Arriba (_aE)
```json
{
  "RV-K023_aE.png": {
    "knee_side": "derecha",
    "image_type": "arriba",
    "landmarks": {
      "TL": [x, y], "TR": [x, y], "BL": [x, y], "BR": [x, y],
      "MC": [x, y], "LC": [x, y], "IG": [x, y]
    },
    "damage_scores": {
      "zona_0": [0, 1, 2, 1, 0, 0, 0], ...  // 10 zonas
    }
  }
}
```

### Imagen Abajo (_bE)
```json
{
  "RV-K023_bE.png": {
    "knee_side": "derecha",
    "image_type": "abajo",
    "landmarks": {
      "AM": [x, y], "AL": [x, y],
      "PM": [x, y], "PL": [x, y]
    },
    "damage_scores": {
      "zona_0": [0, 1, 2, 1, 0, 0, 0], ...  // 4 zonas
    }
  }
}
```

---

## 🧪 Pruebas

Ejemplo creado: `RV-K023_bE.png`
- Estructura: 4 zonas, 7 tipos de daño
- Datos de prueba: Incluye valores ejemplo en damage_scores

Para probar:
1. Abre la app: `streamlit run app.py`
2. Ve a la pestaña "Datos"
3. Selecciona `RV-K023_bE.png`
4. Verás interfaz con 4 puntos en lugar de 10 zonas

---

## 🔄 Flujo de Procesamiento

```
Subir imagen
    ↓
¿Nombre termina en _aE o _bE?
    ↓
├─ _aE (Arriba)
│  ├─ Mostrar 7 landmark buttons
│  ├─ Usar HoodZoneComputer
│  └─ Tabla 10×7
│
└─ _bE (Abajo)
   ├─ Mostrar 4 landmark buttons (AM, AL, PM, PL)
   ├─ Usar FourZoneComputer
   └─ Tabla 4×7
```

---

## ⚙️ Compatibilidad

- ✅ Todas las imágenes _aE existentes funcionan como antes
- ✅ Nuevas imágenes _bE con 4 puntos soportadas
- ✅ Importación automática de tipo según nombre archivo
- ✅ Exportación a annotations.json con estructura correcta

---

## 🚀 Próximos Pasos (Opcional)

1. **Entrenamiento adaptativo**: Modificar `train/dataset.py` para manejar dinámicamente 4 u 10 zonas
2. **Inferencia**: Actualizar `api/inference.py` para detectar automáticamente tipo
3. **Auto-detección de puntos**: Crear detector de los 4 puntos para _bE (similar al de landmarks)
4. **Integración UI**: Mostrar aviso visual ("📷 Vista superior" / "📸 Vista inferior")

---

## 📝 Notas
- Las imágenes son detectadas por el sufijo del nombre archivo: `_aE` vs `_bE`
- Cada tipo tiene su propia geometría de segmentación
- Los scores se normalizan según el número de zonas (4 o 10)
