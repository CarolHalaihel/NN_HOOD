"""
train.py — Bucle de entrenamiento completo para HoodNet

Estrategia: Leave-One-Out Cross-Validation (LOOCV) para dataset < 40 imágenes.
  - Cada fold deja una imagen fuera como validación.
  - Se entrena un HoodNet independiente por fold.
  - Al final, se entrena el modelo final con TODAS las imágenes para producción.

Uso:
  python train.py                             # entrenamiento LOOCV completo (60 épocas)
  python train.py --dry-run                   # verificación rápida (2 épocas, 1 fold)
  python train.py --epochs 40                 # cambiar número de épocas
  python train.py --images-dir ruta/imágenes  # ruta personalizada
  python train.py --skip-loocv                # solo modelo final (más rápido)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Importaciones relativas al paquete hood-backend
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.model import HoodNet
from train.dataset import TibialTrayDataset, build_train_transform, build_val_transform

DAMAGE_TYPES = [
    "delaminacion", "abrasion",    "rayado",      "brunido",
    "picado",       "residuos",    "deformacion", "fatiga",
]


def _write_progress(path, data: dict) -> None:
    """
    Escribe el estado actual del entrenamiento en un archivo JSON.
    La webapp Streamlit lo lee periódicamente para mostrar el progreso.
    Falla silenciosamente para no interrumpir el entrenamiento.
    """
    if not path:
        return
    try:
        Path(path).write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass

# Pesos para CrossEntropyLoss: compensa que score=0 es muy frecuente en implantes poco dañados
# [score_0, score_1, score_2, score_3]
DEFAULT_CLASS_WEIGHTS = torch.tensor([0.5, 1.5, 2.0, 3.0])


# ─────────────────────────────────────────────────────────────────────────────
# MÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(all_preds: list, all_targets: list) -> dict:
    """
    Calcula métricas sobre listas de predicciones y targets acumulados.

    - exact_match    : fracción de muestras donde los 8 scores son exactamente correctos
    - mae_per_damage : error absoluto medio por tipo de daño (métrica ordinal)
    - mae_global     : MAE promedio sobre todos los tipos de daño
    """
    preds   = np.array(all_preds,   dtype=int)  # (N, 8)
    targets = np.array(all_targets, dtype=int)  # (N, 8)

    exact_match = float((preds == targets).all(axis=1).mean())
    mae_per     = np.abs(preds - targets).mean(axis=0)  # (8,)

    mae_dict = {dmg: float(mae_per[i]) for i, dmg in enumerate(DAMAGE_TYPES)}
    mae_dict["_global"] = float(mae_per.mean())

    return {"exact_match": exact_match, "mae_per_damage": mae_dict}


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO DE UNA ÉPOCA
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model: HoodNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.CrossEntropyLoss,
    device: torch.device,
    zone_idx: int,
) -> float:
    """
    Entrena una época del HoodNet sobre la zona 'zone_idx'.

    La imagen completa entra al modelo (el recorte de zona se hace en el dataset
    o aquí se usa la imagen redimensionada directamente — HoodNet recibe la zona
    ya recortada desde el DataLoader).

    Retorna la pérdida media de la época.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        images  = batch["image"].to(device)                               # (B, 3, H, W)
        targets = batch["damage_scores"][:, zone_idx, :].to(device)      # (B, 8)

        optimizer.zero_grad()
        logits = model(images)  # (B, 8, 4)

        # Pérdida: suma de CrossEntropy sobre las 8 cabezas de daño
        loss = sum(
            criterion(logits[:, d, :], targets[:, d])
            for d in range(HoodNet.NUM_DAMAGE)
        )

        loss.backward()
        # Gradient clipping para estabilidad (importante con dataset pequeño)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_zone(
    model: HoodNet,
    loader: DataLoader,
    device: torch.device,
    zone_idx: int,
) -> tuple:
    """
    Evalúa el modelo sobre todas las muestras del loader para la zona 'zone_idx'.
    Retorna (lista_preds, lista_targets) — cada elemento es una lista de 8 scores.
    """
    model.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        images  = batch["image"].to(device)
        targets = batch["damage_scores"][:, zone_idx, :]  # (B, 8) — en CPU

        logits = model(images)                  # (B, 8, 4)
        preds  = logits.argmax(dim=-1).cpu()    # (B, 8)

        all_preds.extend(preds.numpy().tolist())
        all_targets.extend(targets.numpy().tolist())

    return all_preds, all_targets


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO DE UN FOLD LOOCV
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(
    fold_idx: int,
    dataset_full: TibialTrayDataset,
    epochs: int,
    device: torch.device,
    models_dir: Path,
    dry_run: bool = False,
    progress_file: str = None,
    total_folds: int = 1,
) -> dict:
    """
    Entrena un fold LOOCV completo: deja la imagen 'fold_idx' como validación.
    Entrena un HoodNet por cada una de las 10 zonas Hood.

    Retorna dict con métricas por zona: {"zona_0": {"exact_match": ..., ...}, ...}
    """
    n = len(dataset_full)
    train_indices = [i for i in range(n) if i != fold_idx]

    _write_progress(progress_file, {
        "status":        "training",
        "phase":         "LOOCV",
        "current_fold":  fold_idx + 1,
        "total_folds":   total_folds,
        "current_epoch": 0,
        "total_epochs":  2 if dry_run else epochs,
        "zones_done":    0,
    })

    # Subconjuntos: train con augmentación, val sin augmentación
    train_subset = Subset(dataset_full, train_indices)
    val_subset   = Subset(dataset_full, [fold_idx])

    train_loader = DataLoader(
        train_subset, batch_size=8, shuffle=True, num_workers=0, drop_last=False
    )
    val_loader = DataLoader(
        val_subset, batch_size=1, shuffle=False, num_workers=0
    )

    actual_epochs = 2 if dry_run else epochs
    fold_results  = {}

    for zone_idx in range(10):
        # Nuevo modelo y optimizador por zona (cabezas independientes)
        model     = HoodNet().to(device)
        criterion = nn.CrossEntropyLoss(weight=DEFAULT_CLASS_WEIGHTS.to(device))
        optimizer = torch.optim.AdamW(
            model.heads.parameters(), lr=1e-3, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=actual_epochs
        )

        best_val_loss = float("inf")

        for epoch in range(actual_epochs):
            train_one_epoch(model, train_loader, optimizer, criterion, device, zone_idx)
            scheduler.step()
            _write_progress(progress_file, {
                "status":        "training",
                "phase":         "LOOCV",
                "current_fold":  fold_idx + 1,
                "total_folds":   total_folds,
                "current_epoch": epoch + 1,
                "total_epochs":  actual_epochs,
                "zones_done":    zone_idx,
                "current_zone":  zone_idx + 1,
            })

        # Evaluación final del fold en la muestra de validación
        preds, targets = evaluate_zone(model, val_loader, device, zone_idx)
        metrics = compute_metrics(preds, targets)
        fold_results[f"zona_{zone_idx}"] = metrics

    img_name = dataset_full.samples[fold_idx][0]
    em_mean  = np.mean([v["exact_match"] for v in fold_results.values()])
    print(f"  fold {fold_idx+1:02d}/{n} [{img_name}]  exact_match_medio={em_mean:.3f}")

    return fold_results


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO FINAL (con todos los datos — modelo de producción)
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(
    dataset_full: TibialTrayDataset,
    epochs: int,
    device: torch.device,
    models_dir: Path,
    progress_file: str = None,
) -> None:
    """
    Entrena el modelo final usando TODAS las imágenes disponibles.
    Este modelo se exportará a ONNX para el servidor de producción.

    Se entrena UN único HoodNet evaluado sobre TODAS las zonas en cada epoch
    (más eficiente que 10 modelos separados para producción).
    """
    print("\n[INFO] Entrenando modelo final con todo el dataset...")

    loader = DataLoader(
        dataset_full, batch_size=8, shuffle=True, num_workers=0, drop_last=False
    )

    model     = HoodNet().to(device)
    criterion = nn.CrossEntropyLoss(weight=DEFAULT_CLASS_WEIGHTS.to(device))
    optimizer = torch.optim.AdamW(
        model.heads.parameters(), lr=1e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        # Entrenar sobre todas las zonas en cada epoch
        epoch_losses = []
        for zone_idx in range(10):
            loss = train_one_epoch(model, loader, optimizer, criterion, device, zone_idx)
            epoch_losses.append(loss)
        scheduler.step()

        _write_progress(progress_file, {
            "status":        "training",
            "phase":         "final",
            "current_fold":  1,
            "total_folds":   1,
            "current_epoch": epoch + 1,
            "total_epochs":  epochs,
            "loss":          float(np.mean(epoch_losses)),
        })
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} — loss media: {np.mean(epoch_losses):.4f}")

    # Guardar modelo final
    save_path = models_dir / "hoodnet_final.pt"
    torch.save(model.state_dict(), save_path)
    _write_progress(progress_file, {"status": "done", "model_path": str(save_path)})
    print(f"[OK] Modelo final guardado: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento HoodNet con LOOCV — CPU optimizado"
    )
    parser.add_argument(
        "--images-dir", type=str, default="../data/images",
        help="Directorio con imágenes de implantes (relativo a train/)",
    )
    parser.add_argument(
        "--annotations", type=str, default="../data/annotations.json",
        help="Ruta a annotations.json (relativo a train/)",
    )
    parser.add_argument(
        "--models-dir", type=str, default="../models",
        help="Directorio donde guardar modelos .pt (relativo a train/)",
    )
    parser.add_argument(
        "--epochs", type=int, default=60,
        help="Número de épocas de entrenamiento por fold",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Ejecuta solo 2 épocas y 1 fold — verifica que el pipeline funciona",
    )
    parser.add_argument(
        "--skip-loocv", action="store_true",
        help="Omite LOOCV y entrena directamente el modelo final",
    )
    parser.add_argument(
        "--progress-file", type=str, default=None,
        help="Ruta a un archivo JSON donde se escribe el progreso (usado por la webapp)",
    )
    args = parser.parse_args()

    # Resolver rutas relativas al directorio train/
    train_dir        = Path(__file__).resolve().parent
    images_dir       = (train_dir / args.images_dir).resolve()
    annotations_path = (train_dir / args.annotations).resolve()
    models_dir       = (train_dir / args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("HOOD NN — Entrenamiento HoodNet")
    print("=" * 60)
    print(f"  Imágenes     : {images_dir}")
    print(f"  Anotaciones  : {annotations_path}")
    print(f"  Modelos      : {models_dir}")
    print(f"  Épocas       : {args.epochs}")
    print(f"  Dry-run      : {args.dry_run}")
    print(f"  Skip LOOCV   : {args.skip_loocv}")
    print()

    # Verificar existencia de datos
    if not annotations_path.exists():
        print(f"[ERROR] No se encontró annotations.json en:\n  {annotations_path}")
        print("  Ejecuta primero: python tools/import_annotations.py --template")
        sys.exit(1)

    if not images_dir.exists():
        print(f"[ERROR] Directorio de imágenes no encontrado:\n  {images_dir}")
        sys.exit(1)

    # Dispositivo (CPU — sin CUDA)
    device = torch.device("cpu")
    print(f"[INFO] Dispositivo: {device}")

    # Dataset completo con augmentación para entrenamiento
    try:
        dataset_full = TibialTrayDataset(
            images_dir=str(images_dir),
            annotations_path=str(annotations_path),
            augment=True,
            image_size=224,
        )
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    n = len(dataset_full)
    print(f"[INFO] Dataset: {n} imágenes cargadas\n")

    # ── LOOCV ────────────────────────────────────────────────────────────────
    if not args.skip_loocv:
        folds_to_run = list([0] if args.dry_run else range(n))
        print(f"[INFO] Iniciando LOOCV ({len(folds_to_run)} folds)...")
        _write_progress(args.progress_file, {
            "status":      "training",
            "phase":       "LOOCV",
            "total_folds": len(folds_to_run),
            "total_epochs": args.epochs,
        })

        all_fold_results = []
        for fold_idx in folds_to_run:
            fold_results = train_fold(
                fold_idx=fold_idx,
                dataset_full=dataset_full,
                epochs=args.epochs,
                device=device,
                models_dir=models_dir,
                dry_run=args.dry_run,
                progress_file=args.progress_file,
                total_folds=len(folds_to_run),
            )
            all_fold_results.append(fold_results)

        # Resumen LOOCV
        if not args.dry_run:
            print("\n" + "=" * 60)
            print("RESUMEN LOOCV")
            print("=" * 60)
            for zone_idx in range(10):
                key     = f"zona_{zone_idx}"
                ems     = [r[key]["exact_match"] for r in all_fold_results]
                maes    = [r[key]["mae_per_damage"]["_global"] for r in all_fold_results]
                print(
                    f"  {key}: exact_match={np.mean(ems):.3f}±{np.std(ems):.3f}  "
                    f"MAE={np.mean(maes):.3f}"
                )

            # Guardar resultados detallados
            results_path = models_dir / "loocv_results.json"
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(all_fold_results, f, indent=2, ensure_ascii=False)
            print(f"\n[INFO] Resultados LOOCV guardados: {results_path}")

    # ── Modelo final para producción ──────────────────────────────────────────
    if not args.dry_run:
        train_final_model(
            dataset_full=dataset_full,
            epochs=args.epochs,
            device=device,
            models_dir=models_dir,
            progress_file=args.progress_file,
        )
        print("\n[OK] Entrenamiento completado.")
        print("     Siguiente paso: python train/export_onnx.py")
    else:
        print("\n[DRY-RUN OK] El pipeline funciona correctamente.")
        print("             Ejecuta sin --dry-run para entrenamiento completo.")


if __name__ == "__main__":
    main()
