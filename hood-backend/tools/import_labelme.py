"""
import_labelme.py — Convierte anotaciones de LabelMe → coordenadas de landmarks en annotations.json

LabelMe guarda un archivo JSON por imagen cuando usas el modo "point":
  - Marca cada punto anatómico como un punto con etiqueta "TL", "TR", "BL", "BR", "MC", "LC", "IG"
  - Este script lee esos JSON y extrae las coordenadas para cada imagen

INSTRUCCIONES PARA ANOTAR CON LABELME:
  1. pip install labelme
  2. labelme data/images/
  3. Por cada imagen, crear 7 puntos con etiquetas: TL TR BL BR MC LC IG
  4. Guardar → genera un .json por imagen en el mismo directorio
  5. Ejecutar este script

USO:
  python tools/import_labelme.py                         # busca JSON en data/images/
  python tools/import_labelme.py --labelme-dir ruta/     # directorio personalizado
  python tools/import_labelme.py --dry-run               # muestra lo que haría sin guardar
"""

import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

LANDMARK_NAMES = ["TL", "TR", "BL", "BR", "MC", "LC", "IG"]


def parse_labelme_json(json_path: Path) -> dict:
    """
    Lee un archivo JSON de LabelMe y extrae las coordenadas de los 7 landmarks.

    LabelMe guarda puntos como shapes del tipo "point":
      {"label": "TL", "shape_type": "point", "points": [[x, y]]}

    Retorna: {"TL": [x, y], "TR": [x, y], ...} para los landmarks encontrados.
    Retorna {} si el archivo no tiene el formato esperado.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    shapes    = data.get("shapes", [])
    landmarks = {}

    for shape in shapes:
        label      = str(shape.get("label", "")).strip().upper()
        shape_type = shape.get("shape_type", "")
        points     = shape.get("points", [])

        # Solo procesar puntos (no líneas, polígonos, etc.)
        if shape_type != "point":
            continue

        # La etiqueta debe coincidir con uno de los 7 landmarks
        if label not in LANDMARK_NAMES:
            # Intentar alias comunes (p.ej. "top-left" → "TL")
            alias_map = {
                "TOP_LEFT": "TL", "TOP-LEFT": "TL", "TOPLEFT": "TL",
                "TOP_RIGHT": "TR", "TOP-RIGHT": "TR", "TOPRIGHT": "TR",
                "BOT_LEFT": "BL", "BOT-LEFT": "BL", "BOTTOMLEFT": "BL",
                "BOTTOM_LEFT": "BL", "BOTTOM-LEFT": "BL",
                "BOT_RIGHT": "BR", "BOT-RIGHT": "BR", "BOTTOMRIGHT": "BR",
                "BOTTOM_RIGHT": "BR", "BOTTOM-RIGHT": "BR",
                "MEDIAL": "MC", "MEDIAL_CONDYLE": "MC", "MC": "MC",
                "LATERAL": "LC", "LATERAL_CONDYLE": "LC", "LC": "LC",
                "GROOVE": "IG", "INTERCONDYLAR": "IG", "IG": "IG",
            }
            label = alias_map.get(label, "")
            if not label:
                continue

        if points:
            x, y = float(points[0][0]), float(points[0][1])
            landmarks[label] = [x, y]

    return landmarks


def get_image_name_from_labelme(json_path: Path) -> str:
    """
    Extrae el nombre de imagen referenciado en el JSON de LabelMe.
    Si no está disponible, usa el nombre del JSON con extensión .jpg.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    img_path = data.get("imagePath", "")
    if img_path:
        return Path(img_path).name

    # Fallback: mismo nombre que el JSON pero con .jpg
    return json_path.with_suffix(".jpg").name


def import_labelme_dir(
    labelme_dir: Path,
    annotations_file: Path,
    dry_run: bool = False,
) -> dict:
    """
    Escanea un directorio buscando archivos JSON de LabelMe y actualiza
    annotations.json con las coordenadas de landmarks extraídas.

    Solo actualiza el campo "landmarks" — no toca los damage_scores existentes.

    Retorna dict con resumen: {"updated": [...], "missing_landmarks": {...}, "skipped": [...]}
    """
    json_files = sorted(labelme_dir.glob("*.json"))

    if not json_files:
        return {"updated": [], "missing_landmarks": {}, "skipped": [], "error": "No se encontraron archivos .json"}

    # Cargar anotaciones existentes
    annotations = {}
    if annotations_file.exists():
        try:
            with open(annotations_file, "r", encoding="utf-8") as f:
                all_data = json.load(f)
            # Filtrar claves internas
            annotations = {k: v for k, v in all_data.items() if not k.startswith("_")}
        except Exception:
            pass

    updated           = []
    missing_landmarks = {}
    skipped           = []

    for json_path in json_files:
        # Ignorar annotations.json si estuviera en el mismo directorio
        if json_path.name == "annotations.json":
            continue

        try:
            img_name  = get_image_name_from_labelme(json_path)
            landmarks = parse_labelme_json(json_path)

            if not landmarks:
                skipped.append(json_path.name)
                continue

            # Detectar landmarks faltantes
            missing = [lm for lm in LANDMARK_NAMES if lm not in landmarks]
            if missing:
                missing_landmarks[img_name] = missing

            # Inicializar anotación si la imagen es nueva
            if img_name not in annotations:
                annotations[img_name] = {
                    "landmarks":     {lm: [0.0, 0.0] for lm in LANDMARK_NAMES},
                    "damage_scores": {f"zona_{z}": [0] * 7 for z in range(10)},
                }

            # Actualizar solo los landmarks encontrados
            annotations[img_name]["landmarks"].update(landmarks)
            updated.append(img_name)

        except Exception as ex:
            skipped.append(f"{json_path.name} ({ex})")

    if not dry_run and updated:
        # Preservar claves internas del archivo original
        original_internal = {}
        if annotations_file.exists():
            try:
                with open(annotations_file, "r", encoding="utf-8") as f:
                    orig = json.load(f)
                original_internal = {k: v for k, v in orig.items() if k.startswith("_")}
            except Exception:
                pass

        combined = {**annotations, **original_internal}
        annotations_file.parent.mkdir(parents=True, exist_ok=True)
        with open(annotations_file, "w", encoding="utf-8") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)

    return {
        "updated":           updated,
        "missing_landmarks": missing_landmarks,
        "skipped":           skipped,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Importar landmarks de LabelMe (.json) → annotations.json"
    )
    parser.add_argument(
        "--labelme-dir", type=str, default="../data/images",
        help="Directorio donde LabelMe guardó los .json (por defecto: data/images/)",
    )
    parser.add_argument(
        "--output", type=str, default="../data/annotations.json",
        help="Ruta al annotations.json a actualizar",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Muestra los resultados sin modificar annotations.json",
    )
    args = parser.parse_args()

    tools_dir     = Path(__file__).resolve().parent
    labelme_dir   = (tools_dir / args.labelme_dir).resolve()
    ann_file      = (tools_dir / args.output).resolve()

    print(f"[INFO] Buscando JSON de LabelMe en: {labelme_dir}")
    print(f"[INFO] Actualizando: {ann_file}")
    if args.dry_run:
        print("[INFO] Modo DRY-RUN: no se escribirá nada\n")

    result = import_labelme_dir(labelme_dir, ann_file, dry_run=args.dry_run)

    if result.get("error"):
        print(f"[ERROR] {result['error']}")
        sys.exit(1)

    print(f"[OK] Imágenes actualizadas: {len(result['updated'])}")
    for name in result["updated"]:
        ml = result["missing_landmarks"].get(name)
        tag = f"  (faltan: {ml})" if ml else ""
        print(f"      {name}{tag}")

    if result["skipped"]:
        print(f"\n[AVISO] Ignorados: {len(result['skipped'])}")
        for s in result["skipped"]:
            print(f"      {s}")

    if result["missing_landmarks"]:
        print(f"\n[AVISO] Imágenes con landmarks incompletos:")
        for img, missing in result["missing_landmarks"].items():
            print(f"      {img}: faltan {missing}")

    if not args.dry_run and result["updated"]:
        print(f"\n[OK] annotations.json actualizado: {ann_file}")
        print("     Los damage_scores no han sido modificados.")
        print("     Siguiente: añade los scores desde Excel o la webapp.")


if __name__ == "__main__":
    main()
