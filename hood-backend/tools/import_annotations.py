"""
import_annotations.py — Importa anotaciones desde tabla externa → annotations.json

Convierte una hoja de cálculo (Excel .xlsx o CSV) con los datos Hood al formato
JSON requerido por TibialTrayDataset.

FORMATOS SOPORTADOS:

  Formato WIDE (recomendado, --format wide):
    Una fila por imagen. Columnas:
      imagen, landmark_TL_x, landmark_TL_y, ..., landmark_IG_y (14 cols de landmarks)
      zona_0_delaminacion, ..., zona_9_fatiga (80 cols de scores)

  Formato LONG (--format long):
    Una fila por (imagen, zona). Columnas:
      imagen, zona_idx (0-9), delaminacion, abrasion, ..., fatiga
    Los landmarks pueden estar en columnas opcionales (solo en la fila zona_idx=0).

USO:
  # 1. Generar plantilla Excel vacía para rellenar
  python tools/import_annotations.py --template

  # 2. Importar desde Excel formato wide
  python tools/import_annotations.py --input mis_datos.xlsx

  # 3. Importar desde CSV formato long
  python tools/import_annotations.py --input datos.csv --format long

  # 4. Especificar hoja Excel y salida personalizada
  python tools/import_annotations.py --input libro.xlsx --sheet Hoja2 --output mi_anotaciones.json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

LANDMARK_NAMES = ["TL", "TR", "BL", "BR", "MC", "LC", "IG"]

DAMAGE_TYPES = [
    "delaminacion", "abrasion",    "rayado",      "brunido",
    "picado",       "residuos",    "deformacion", "fatiga",
]

NUM_ZONES   = 10
NUM_DAMAGES = 8


# ─────────────────────────────────────────────────────────────────────────────
# GENERADOR DE PLANTILLA
# ─────────────────────────────────────────────────────────────────────────────

def generate_template(output_path: Path) -> None:
    """
    Genera una plantilla Excel vacía lista para rellenar manualmente.
    Incluye una fila de ejemplo con formato correcto.
    """
    # Columna de nombre de imagen
    columns = ["imagen"]

    # Columnas de landmarks (14 columnas: x e y por cada uno de los 7 landmarks)
    for lm in LANDMARK_NAMES:
        columns += [f"landmark_{lm}_x", f"landmark_{lm}_y"]

    # Columnas de scores Hood (80 columnas: 10 zonas × 8 tipos de daño)
    for z in range(NUM_ZONES):
        for dmg in DAMAGE_TYPES:
            columns.append(f"zona_{z}_{dmg}")

    # Fila de ejemplo: landmarks de una imagen 512×512 típica
    example_row = ["imagen_001.jpg"]
    example_row += [
        120, 45,   # TL
        390, 45,   # TR
        120, 470,  # BL
        390, 470,  # BR
        220, 180,  # MC
        310, 175,  # LC
        265, 200,  # IG
    ]
    example_row += [0] * (NUM_ZONES * NUM_DAMAGES)  # todos los scores a 0

    df = pd.DataFrame([example_row], columns=columns)

    # Añadir fila de descripción como segunda fila (comentario)
    desc_row = ["EJEMPLO — reemplaza con tus datos"]
    desc_row += ["coord_x", "coord_y"] * len(LANDMARK_NAMES)
    desc_row += ["0-3"] * (NUM_ZONES * NUM_DAMAGES)
    df_desc = pd.DataFrame([desc_row], columns=columns)

    df_final = pd.concat([df, df_desc], ignore_index=True)
    df_final.to_excel(output_path, index=False)

    print(f"[OK] Plantilla generada: {output_path}")
    print(f"     Columnas totales: {len(columns)}")
    print(f"       - 1 col imagen")
    print(f"       - {len(LANDMARK_NAMES) * 2} cols landmarks")
    print(f"       - {NUM_ZONES * NUM_DAMAGES} cols scores Hood")
    print()
    print("     Rellena las filas (una por imagen) y luego ejecuta:")
    print(f"     python tools/import_annotations.py --input {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTACIÓN FORMATO WIDE
# ─────────────────────────────────────────────────────────────────────────────

def import_wide_format(df: pd.DataFrame) -> dict:
    """
    Importa tabla wide: cada fila = una imagen completa con todas las zonas.

    Columnas esperadas:
      imagen, landmark_TL_x, landmark_TL_y, ..., zona_0_delaminacion, ..., zona_9_fatiga
    """
    annotations = {}
    errors = []

    for row_idx, row in df.iterrows():
        img_name = str(row.get("imagen", "")).strip()
        if not img_name or img_name.lower() in ("nan", "none", ""):
            continue

        # ── Landmarks ────────────────────────────────────────────────────────
        landmarks = {}
        for lm in LANDMARK_NAMES:
            x_col = f"landmark_{lm}_x"
            y_col = f"landmark_{lm}_y"
            try:
                lx = float(row[x_col]) if x_col in row.index else 0.0
                ly = float(row[y_col]) if y_col in row.index else 0.0
                if pd.isna(lx) or pd.isna(ly):
                    raise ValueError("NaN")
                landmarks[lm] = [lx, ly]
            except (ValueError, TypeError, KeyError):
                landmarks[lm] = [0.0, 0.0]
                errors.append(f"Fila {row_idx}: landmark {lm} inválido para '{img_name}'")

        # ── Damage scores ─────────────────────────────────────────────────────
        damage_scores = {}
        for z in range(NUM_ZONES):
            zone_key = f"zona_{z}"
            scores   = []
            for dmg in DAMAGE_TYPES:
                col = f"zona_{z}_{dmg}"
                try:
                    val = int(float(row[col])) if col in row.index else 0
                    val = max(0, min(3, val))  # clamp a rango válido
                    if pd.isna(float(row.get(col, 0))):
                        val = 0
                except (ValueError, TypeError):
                    val = 0
                scores.append(val)
            damage_scores[zone_key] = scores

        annotations[img_name] = {
            "landmarks":     landmarks,
            "damage_scores": damage_scores,
        }

    if errors:
        print(f"[AVISO] {len(errors)} advertencias durante la importación:")
        for e in errors[:10]:  # mostrar máximo 10
            print(f"  - {e}")
        if len(errors) > 10:
            print(f"  ... y {len(errors) - 10} más.")

    return annotations


# ─────────────────────────────────────────────────────────────────────────────
# IMPORTACIÓN FORMATO LONG
# ─────────────────────────────────────────────────────────────────────────────

def import_long_format(df: pd.DataFrame) -> dict:
    """
    Importa tabla long: cada fila = una zona de una imagen.

    Columnas requeridas: imagen, zona_idx, + 8 columnas de tipos de daño
    Columnas opcionales: landmark_TL_x, ..., landmark_IG_y (en filas con zona_idx=0)
    """
    required = ["imagen", "zona_idx"] + DAMAGE_TYPES
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columnas faltantes en formato long: {missing}\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    annotations: dict = {}

    for _, row in df.iterrows():
        img_name = str(row["imagen"]).strip()
        if not img_name or img_name.lower() in ("nan", "none"):
            continue

        try:
            zone_idx = int(float(row["zona_idx"]))
        except (ValueError, TypeError):
            continue

        if zone_idx < 0 or zone_idx >= NUM_ZONES:
            continue

        # Inicializar entrada si es la primera zona de esta imagen
        if img_name not in annotations:
            annotations[img_name] = {
                "landmarks":     {lm: [0.0, 0.0] for lm in LANDMARK_NAMES},
                "damage_scores": {f"zona_{z}": [0] * NUM_DAMAGES for z in range(NUM_ZONES)},
            }

        # Scores de esta zona
        scores = []
        for dmg in DAMAGE_TYPES:
            try:
                val = int(float(row[dmg]))
                val = max(0, min(3, val))
                if pd.isna(float(row[dmg])):
                    val = 0
            except (ValueError, TypeError):
                val = 0
            scores.append(val)
        annotations[img_name]["damage_scores"][f"zona_{zone_idx}"] = scores

        # Landmarks opcionales (solo desde la primera zona para evitar sobreescritura)
        if zone_idx == 0:
            for lm in LANDMARK_NAMES:
                x_col, y_col = f"landmark_{lm}_x", f"landmark_{lm}_y"
                if x_col in row.index and y_col in row.index:
                    try:
                        lx = float(row[x_col])
                        ly = float(row[y_col])
                        if not (pd.isna(lx) or pd.isna(ly)):
                            annotations[img_name]["landmarks"][lm] = [lx, ly]
                    except (ValueError, TypeError):
                        pass

    return annotations


# ─────────────────────────────────────────────────────────────────────────────
# VALIDACIÓN DEL JSON GENERADO
# ─────────────────────────────────────────────────────────────────────────────

def validate_annotations(annotations: dict) -> None:
    """
    Valida la estructura del JSON generado.
    Imprime advertencias si hay problemas pero no detiene el proceso.
    """
    issues = 0
    for img_name, ann in annotations.items():
        # Verificar landmarks
        for lm in LANDMARK_NAMES:
            if lm not in ann.get("landmarks", {}):
                print(f"[AVISO] {img_name}: falta landmark '{lm}'")
                issues += 1

        # Verificar scores
        for z in range(NUM_ZONES):
            key    = f"zona_{z}"
            scores = ann.get("damage_scores", {}).get(key, [])
            if len(scores) != NUM_DAMAGES:
                print(
                    f"[AVISO] {img_name}: zona_{z} tiene {len(scores)} scores "
                    f"(esperados {NUM_DAMAGES})"
                )
                issues += 1
            for s in scores:
                if s not in (0, 1, 2, 3):
                    print(f"[AVISO] {img_name}: zona_{z} tiene score inválido: {s}")
                    issues += 1

    if issues == 0:
        print("[OK] Validación: sin problemas encontrados.")
    else:
        print(f"[AVISO] Validación: {issues} problema(s) encontrado(s). Revisa las advertencias.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Importar anotaciones Hood desde Excel/CSV → annotations.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", type=str,
        help="Ruta al archivo Excel (.xlsx) o CSV con las anotaciones",
    )
    parser.add_argument(
        "--output", type=str, default="../data/annotations.json",
        help="Ruta de salida para annotations.json (relativo a tools/)",
    )
    parser.add_argument(
        "--format", choices=["wide", "long"], default="wide",
        help="Formato de la tabla: 'wide' (una fila/imagen) o 'long' (una fila/zona)",
    )
    parser.add_argument(
        "--sheet", type=str, default=None,
        help="Nombre de la hoja Excel (por defecto: primera hoja)",
    )
    parser.add_argument(
        "--template", action="store_true",
        help="Genera plantilla Excel vacía y termina",
    )
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Omitir validación del JSON generado",
    )
    args = parser.parse_args()

    tools_dir = Path(__file__).resolve().parent

    # ── Modo plantilla ────────────────────────────────────────────────────────
    if args.template:
        template_path = tools_dir / "plantilla_anotaciones.xlsx"
        generate_template(template_path)
        return

    # ── Importación ───────────────────────────────────────────────────────────
    if not args.input:
        parser.print_help()
        print("\n[ERROR] Especifica --input <archivo> o usa --template")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (tools_dir / input_path).resolve()

    if not input_path.exists():
        print(f"[ERROR] Archivo no encontrado: {input_path}")
        sys.exit(1)

    # Cargar tabla
    print(f"[INFO] Cargando: {input_path}")
    try:
        if input_path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(input_path, sheet_name=args.sheet)
        elif input_path.suffix.lower() == ".csv":
            df = pd.read_csv(input_path)
        else:
            print(f"[ERROR] Extensión no soportada: {input_path.suffix}. Use .xlsx o .csv")
            sys.exit(1)
    except Exception as exc:
        print(f"[ERROR] No se pudo leer el archivo: {exc}")
        sys.exit(1)

    # Eliminar filas completamente vacías
    df = df.dropna(how="all").reset_index(drop=True)
    print(f"[INFO] Filas: {len(df)}, Columnas: {len(df.columns)}")

    # Importar según formato
    try:
        if args.format == "wide":
            annotations = import_wide_format(df)
        else:
            annotations = import_long_format(df)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    print(f"[INFO] Imágenes importadas: {len(annotations)}")

    # Validar estructura
    if not args.no_validate:
        validate_annotations(annotations)

    # Guardar JSON
    output_path = (tools_dir / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] annotations.json guardado: {output_path}")

    # Estadísticas de resumen
    total_scores = sum(
        score
        for ann in annotations.values()
        for zone_scores in ann["damage_scores"].values()
        for score in zone_scores
    )
    max_possible = len(annotations) * NUM_ZONES * NUM_DAMAGES * 3
    print(f"     Imágenes : {len(annotations)}")
    print(f"     Score total acumulado : {total_scores} / {max_possible} posibles")
    print(f"\n[INFO] Siguiente paso: python train/train.py --dry-run")


if __name__ == "__main__":
    main()
