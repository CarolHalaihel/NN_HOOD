#!/usr/bin/env python3
"""
Script para eliminar la columna "Fatiga" de todos los damage_scores en annotations.json
"""
import json
from pathlib import Path

# Rutas
ROOT = Path(__file__).resolve().parent
ANNOTATIONS_FILE = ROOT / "data" / "annotations.json"

# Cargar anotaciones
with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Contar cambios
total_images = len(data)
total_zones_modified = 0

# Procesar cada imagen
for image_name, image_data in data.items():
    if "damage_scores" in image_data:
        damage_scores = image_data["damage_scores"]
        
        # Procesar cada zona
        for zona_name, scores in damage_scores.items():
            # El array tiene 8 elementos, eliminar el último (índice 7 = Fatiga)
            if isinstance(scores, list) and len(scores) == 8:
                # Mantener solo los primeros 7 elementos
                damage_scores[zona_name] = scores[:7]
                total_zones_modified += 1

# Guardar anotaciones actualizadas
with open(ANNOTATIONS_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✅ Columna 'Fatiga' eliminada exitosamente")
print(f"   - Total imágenes procesadas: {total_images}")
print(f"   - Total zonas modificadas: {total_zones_modified}")
print(f"   - Archivo guardado: {ANNOTATIONS_FILE}")
