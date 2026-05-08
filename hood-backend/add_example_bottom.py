#!/usr/bin/env python3
"""
Script para añadir un ejemplo de imagen _bE (abajo) con 4 zonas a annotations.json
"""
import json
from pathlib import Path

# Rutas
ROOT = Path(__file__).resolve().parent
ANNOTATIONS_FILE = ROOT / "data" / "annotations.json"

# Cargar anotaciones
with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Crear ejemplo de imagen de abajo (parte inferior de la prótesis)
example_bottom = {
    "knee_side":     "derecha",
    "image_type":    "abajo",
    "landmarks": {
        "AM": [200, 150],   # Anteromedial (arriba-izquierda)
        "AL": [400, 150],   # Anterolateral (arriba-derecha)
        "PM": [200, 350],   # Posteromedial (abajo-izquierda)
        "PL": [400, 350],   # Posterolateral (abajo-derecha)
    },
    "damage_scores": {
        "zona_0": [0, 0, 1, 0, 1, 0, 0],  # AM
        "zona_1": [0, 1, 0, 0, 0, 0, 0],  # AL
        "zona_2": [0, 0, 0, 0, 0, 0, 0],  # PM
        "zona_3": [0, 0, 1, 0, 0, 0, 0],  # PL
    }
}

# Añadir ejemplo (sin sobrescribir imágenes existentes)
example_name = "RV-K023_bE.png"
if example_name not in data:
    data[example_name] = example_bottom
    print(f"✅ Añadido ejemplo: {example_name}")
else:
    print(f"⚠️ {example_name} ya existe. Actualizando...")
    data[example_name] = example_bottom

# Guardar
with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"📁 Archivo guardado: {ANNOTATIONS_FILE}")
print(f"Total imágenes: {len(data)}")
