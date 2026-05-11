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
from torch.utils.data import DataLoader

# Importaciones relativas al paquete hood-backend
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.model import HoodNet
from train.dataset import ZoneCropDataset, load_base_samples, build_train_transform, build_val_transform

DAMAGE_TYPES = [
    "rayado",       "picado",      "brunido",     "abrasion",
    "delaminacion", "deformacion", "residuos",
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


def _save_checkpoint(checkpoint_file, data: dict) -> None:
    """Guarda el estado de entrenamiento en disco para poder reanudarlo."""
    if not checkpoint_file:
        return
    try:
        torch.save(data, str(checkpoint_file))
    except Exception as e:
        print(f"[WARN] No se pudo guardar checkpoint: {e}", flush=True)


def _load_checkpoint(checkpoint_file) -> dict:
    """Carga el checkpoint previo. Retorna None si no existe o hay error."""
    if not checkpoint_file:
        return None
    p = Path(checkpoint_file)
    if not p.exists():
        return None
    try:
        return torch.load(str(p), weights_only=False)
    except Exception as e:
        print(f"[WARN] No se pudo cargar checkpoint: {e}", flush=True)
        return None


def _should_stop(stop_file) -> bool:
    """Retorna True si la webapp solicitó parada escribiendo el fichero bandera."""
    if not stop_file:
        return False
    return Path(stop_file).exists()

# Pesos para CrossEntropyLoss: compensa que score=0 es muy frecuente en implantes poco dañados
# [score_0, score_1, score_2, score_3]
DEFAULT_CLASS_WEIGHTS = torch.tensor([0.5, 1.5, 2.0, 3.0])

# Hiperparámetros de entrenamiento optimizados para dataset pequeño (≈13 muestras)
# weight_decay=5e-3  : regularización L2 moderada-alta — evita overfitting sin
#                      competir demasiado con el gradiente (1e-2 causaba oscilación)
# label_smoothing=0.05: suaviza objetivos levemente — 0.1 subía demasiado el mínimo
#                       teórico de la loss (~×7 cabezas) haciendo la curva ilegible
LR              = 1e-3
WEIGHT_DECAY    = 5e-3
LABEL_SMOOTHING = 0.05


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
) -> float:
    """
    Entrena una época del HoodNet sobre recortes de zona.
    Cada muestra ya es un recorte de una zona específica con sus 7 scores.
    Retorna la pérdida media de la época.
    """
    model.train()
    total_loss = 0.0

    for batch in loader:
        images  = batch["image"].to(device)           # (B, 3, H, W)
        targets = batch["damage_scores"].to(device)   # (B, 7)

        optimizer.zero_grad()
        logits = model(images)  # (B, 7, 4)

        loss = sum(
            criterion(logits[:, d, :], targets[:, d])
            for d in range(HoodNet.NUM_DAMAGE)
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate_zone(
    model: HoodNet,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Evalúa el modelo sobre recortes de zona.
    Retorna (lista_preds, lista_targets) donde cada elemento tiene 7 scores.
    """
    model.eval()
    all_preds, all_targets = [], []

    for batch in loader:
        images  = batch["image"].to(device)
        targets = batch["damage_scores"]   # (B, 7) en CPU

        logits = model(images)                  # (B, 7, 4)
        preds  = logits.argmax(dim=-1).cpu()    # (B, 7)

        all_preds.extend(preds.numpy().tolist())
        all_targets.extend(targets.numpy().tolist())

    return all_preds, all_targets


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO DE UN FOLD LOOCV
# ─────────────────────────────────────────────────────────────────────────────

def train_fold(
    fold_idx: int,
    base_samples: list,
    images_dir: Path,
    epochs: int,
    device: torch.device,
    models_dir: Path,
    dry_run: bool = False,
    progress_file: str = None,
    total_folds: int = 1,
    checkpoint_file: str = None,
    stop_file: str = None,
    resume_zone: int = 0,
    resume_epoch: int = 0,
    resume_heads_state: dict = None,
    resume_optimizer_state: dict = None,
    resume_scheduler_state: dict = None,
    fold_results_partial: dict = None,
    all_fold_results_so_far: list = None,
) -> tuple:
    """
    Entrena un fold LOOCV completo: deja la imagen 'fold_idx' como validación.
    Soporta parada graceful (stop_file) y reanudación (resume_*).

    Retorna (fold_results, stopped_early):
      - fold_results   : dict por zona {"zona_0": {metrics}, ...}
      - stopped_early  : True si se interrumpió por stop_file
    """
    n             = len(base_samples)
    train_samples = [base_samples[i] for i in range(n) if i != fold_idx]
    val_samples   = [base_samples[fold_idx]]

    actual_epochs = 2 if dry_run else epochs
    fold_results  = dict(fold_results_partial) if fold_results_partial else {}

    # Parámetros de reanudación activos solo para el primer fold que procesamos
    _resume_zone  = resume_zone
    _resume_epoch = resume_epoch
    _heads_state  = resume_heads_state
    _opt_state    = resume_optimizer_state
    _sched_state  = resume_scheduler_state

    for zone_idx in range(10):
        # Saltar zonas ya completadas (sus resultados están en fold_results_partial)
        if zone_idx < _resume_zone:
            continue

        # Verificar parada antes de empezar una nueva zona
        if _should_stop(stop_file):
            _save_checkpoint(checkpoint_file, {
                "version": 1,
                "phase": "loocv",
                "epochs": epochs,
                "dry_run": dry_run,
                "all_fold_results": all_fold_results_so_far or [],
                "current_fold_idx": fold_idx,
                "fold_results_partial": fold_results,
                "current_zone_idx": zone_idx,
                "completed_epochs": 0,
                "heads_state": None,
                "optimizer_state": None,
                "scheduler_state": None,
            })
            _write_progress(progress_file, {
                "status": "paused",
                "phase": "LOOCV",
                "current_fold": fold_idx + 1,
                "total_folds": total_folds,
                "current_epoch": 0,
                "total_epochs": actual_epochs,
                "current_zone": zone_idx + 1,
            })
            print(f"[PAUSED] Checkpoint guardado (fold {fold_idx+1}, zona {zone_idx+1})",
                  flush=True)
            return fold_results, True

        _write_progress(progress_file, {
            "status":        "training",
            "phase":         "LOOCV",
            "current_fold":  fold_idx + 1,
            "total_folds":   total_folds,
            "current_epoch": _resume_epoch if zone_idx == _resume_zone else 0,
            "total_epochs":  actual_epochs,
            "zones_done":    len(fold_results),
            "current_zone":  zone_idx + 1,
        })

        # Dataset de recortes específico para esta zona
        train_ds = ZoneCropDataset(
            images_dir=str(images_dir),
            base_samples=train_samples,
            zone_idx=zone_idx,
            augment=True,
            image_size=224,
            repeat_factor=8,
        )
        val_ds = ZoneCropDataset(
            images_dir=str(images_dir),
            base_samples=val_samples,
            zone_idx=zone_idx,
            augment=False,
            image_size=224,
            repeat_factor=1,
        )

        train_loader = DataLoader(
            train_ds, batch_size=8, shuffle=True, num_workers=0, drop_last=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=0
        )

        # Nuevo modelo y optimizador por zona (cabezas independientes)
        model     = HoodNet().to(device)
        criterion = nn.CrossEntropyLoss(
            weight=DEFAULT_CLASS_WEIGHTS.to(device),
            label_smoothing=LABEL_SMOOTHING,
        )
        optimizer = torch.optim.AdamW(
            model.heads.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=actual_epochs
        )

        # Restaurar estado si es la zona interrumpida
        epoch_start = 0
        if zone_idx == _resume_zone and _heads_state is not None:
            model.heads.load_state_dict(_heads_state)
            optimizer.load_state_dict(_opt_state)
            scheduler.load_state_dict(_sched_state)
            epoch_start = _resume_epoch
            print(f"  [resumiendo] fold {fold_idx+1}, zona {zone_idx+1} "
                  f"desde época {epoch_start+1}", flush=True)

        ep_label = f"{epoch_start+1}..{actual_epochs}" if epoch_start else str(actual_epochs)
        print(f"  [fold {fold_idx+1}/{n}] zona {zone_idx+1}/10 — {ep_label} épocas...",
              flush=True)

        stopped_in_zone = False
        for epoch in range(epoch_start, actual_epochs):
            # Verificar parada al inicio de cada época
            if _should_stop(stop_file):
                _save_checkpoint(checkpoint_file, {
                    "version": 1,
                    "phase": "loocv",
                    "epochs": epochs,
                    "dry_run": dry_run,
                    "all_fold_results": all_fold_results_so_far or [],
                    "current_fold_idx": fold_idx,
                    "fold_results_partial": fold_results,
                    "current_zone_idx": zone_idx,
                    "completed_epochs": epoch,
                    "heads_state": model.heads.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                })
                _write_progress(progress_file, {
                    "status": "paused",
                    "phase": "LOOCV",
                    "current_fold": fold_idx + 1,
                    "total_folds": total_folds,
                    "current_epoch": epoch,
                    "total_epochs": actual_epochs,
                    "current_zone": zone_idx + 1,
                })
                print(f"[PAUSED] Checkpoint guardado "
                      f"(fold {fold_idx+1}, zona {zone_idx+1}, época {epoch})",
                      flush=True)
                stopped_in_zone = True
                break

            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            scheduler.step()
            print(f"    época {epoch+1:3d}/{actual_epochs}  loss={loss:.4f}", flush=True)
            _write_progress(progress_file, {
                "status":        "training",
                "phase":         "LOOCV",
                "current_fold":  fold_idx + 1,
                "total_folds":   total_folds,
                "current_epoch": epoch + 1,
                "total_epochs":  actual_epochs,
                "zones_done":    len(fold_results),
                "current_zone":  zone_idx + 1,
            })
            # Guardar checkpoint tras cada época completada
            _save_checkpoint(checkpoint_file, {
                "version": 1,
                "phase": "loocv",
                "epochs": epochs,
                "dry_run": dry_run,
                "all_fold_results": all_fold_results_so_far or [],
                "current_fold_idx": fold_idx,
                "fold_results_partial": fold_results,
                "current_zone_idx": zone_idx,
                "completed_epochs": epoch + 1,
                "heads_state": model.heads.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            })

        if stopped_in_zone:
            return fold_results, True

        # Evaluación final de la zona en la muestra de validación
        preds, targets = evaluate_zone(model, val_loader, device)
        metrics = compute_metrics(preds, targets)
        fold_results[f"zona_{zone_idx}"] = metrics
        em_zone = np.mean([metrics["exact_match"]] if isinstance(metrics["exact_match"], float)
                          else metrics["exact_match"])
        print(f"  [fold {fold_idx+1}/{n}] zona {zone_idx+1}/10 — exact_match={em_zone:.3f}",
              flush=True)

        # Tras completar la primera zona reanudada, resetear parámetros de reanudación
        _resume_zone  = zone_idx + 1
        _resume_epoch = 0
        _heads_state  = None
        _opt_state    = None
        _sched_state  = None

    img_name = base_samples[fold_idx][0]
    em_mean  = np.mean([v["exact_match"] for v in fold_results.values()])
    print(f"  fold {fold_idx+1:02d}/{n} [{img_name}]  exact_match_medio={em_mean:.3f}",
          flush=True)

    return fold_results, False


# ─────────────────────────────────────────────────────────────────────────────
# ENTRENAMIENTO FINAL (con todos los datos — modelo de producción)
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(
    base_samples: list,
    images_dir: Path,
    epochs: int,
    device: torch.device,
    models_dir: Path,
    progress_file: str = None,
    checkpoint_file: str = None,
    stop_file: str = None,
    start_epoch: int = 0,
    heads_state: dict = None,
    optimizer_state: dict = None,
    scheduler_state: dict = None,
    all_fold_results: list = None,
) -> bool:
    """
    Entrena el modelo final usando TODAS las imágenes disponibles.
    Soporta parada graceful y reanudación desde una época concreta.

    Retorna True si se detuvo antes de completar.
    """
    print("\n[INFO] Entrenando modelo final con todo el dataset...")

    # Pre-crear datasets/loaders para las 10 zonas (fuera del bucle de épocas)
    zone_loaders = []
    for zone_idx in range(10):
        ds = ZoneCropDataset(
            images_dir=str(images_dir),
            base_samples=base_samples,
            zone_idx=zone_idx,
            augment=True,
            image_size=224,
            repeat_factor=8,
        )
        zone_loaders.append(
            DataLoader(ds, batch_size=8, shuffle=True, num_workers=0, drop_last=False)
        )

    model     = HoodNet().to(device)
    criterion = nn.CrossEntropyLoss(
        weight=DEFAULT_CLASS_WEIGHTS.to(device),
        label_smoothing=LABEL_SMOOTHING,
    )
    optimizer = torch.optim.AdamW(
        model.heads.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Restaurar estado si se reanuda
    if heads_state is not None:
        model.heads.load_state_dict(heads_state)
        optimizer.load_state_dict(optimizer_state)
        scheduler.load_state_dict(scheduler_state)
        print(f"[resumiendo] Modelo final desde época {start_epoch+1}", flush=True)

    for epoch in range(start_epoch, epochs):
        # Verificar parada al inicio de cada época
        if _should_stop(stop_file):
            _save_checkpoint(checkpoint_file, {
                "version": 1,
                "phase": "final",
                "epochs": epochs,
                "completed_epochs": epoch,
                "heads_state": model.heads.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "all_fold_results": all_fold_results or [],
            })
            _write_progress(progress_file, {
                "status": "paused",
                "phase": "final",
                "current_epoch": epoch,
                "total_epochs": epochs,
            })
            print(f"[PAUSED] Checkpoint guardado (modelo final, época {epoch})", flush=True)
            return True

        epoch_losses = []
        for loader in zone_loaders:
            loss = train_one_epoch(model, loader, optimizer, criterion, device)
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
            print(f"  Epoch {epoch+1:3d}/{epochs} — loss media: {np.mean(epoch_losses):.4f}",
                  flush=True)

        # Guardar checkpoint tras cada época completada
        _save_checkpoint(checkpoint_file, {
            "version": 1,
            "phase": "final",
            "epochs": epochs,
            "completed_epochs": epoch + 1,
            "heads_state": model.heads.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "all_fold_results": all_fold_results or [],
        })

    save_path = models_dir / "hoodnet_final.pt"
    torch.save(model.state_dict(), save_path)
    # Borrar checkpoint al completar
    if checkpoint_file:
        try:
            Path(checkpoint_file).unlink(missing_ok=True)
        except Exception:
            pass
    _write_progress(progress_file, {"status": "done", "model_path": str(save_path)})
    print(f"[OK] Modelo final guardado: {save_path}")
    return False


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
    parser.add_argument(
        "--checkpoint-file", type=str, default=None,
        help="Ruta al archivo .pt de checkpoint para pausar/reanudar el entrenamiento",
    )
    parser.add_argument(
        "--stop-file", type=str, default=None,
        help="Fichero bandera: si existe, el entrenamiento se pausa y guarda checkpoint",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Reanudar el entrenamiento desde el checkpoint indicado por --checkpoint-file",
    )
    args = parser.parse_args()

    # Forzar line-buffering en stdout para que los print() aparezcan
    # inmediatamente en el log (cuando stdout no es una TTY, Python
    # usa full-buffering por defecto y los mensajes se acumulan en memoria).
    sys.stdout.reconfigure(line_buffering=True)

    # Resolver rutas relativas al directorio train/
    train_dir        = Path(__file__).resolve().parent
    images_dir       = (train_dir / args.images_dir).resolve()
    annotations_path = (train_dir / args.annotations).resolve()
    models_dir       = (train_dir / args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = str((Path(args.checkpoint_file)).resolve()) if args.checkpoint_file else None
    stop_file       = str((Path(args.stop_file)).resolve())       if args.stop_file       else None

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

    # Cargar lista base de imágenes únicas con landmarks completos
    try:
        base_samples = load_base_samples(
            images_dir=images_dir,
            annotations_path=annotations_path,
            only_aE=True,
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    if not base_samples:
        print("[ERROR] No se encontraron imágenes válidas con landmarks completos.")
        sys.exit(1)

    n = len(base_samples)
    print(f"[INFO] Imágenes únicas con landmarks: {n}")
    print(f"[INFO] Muestras de recorte por época (10 zonas × {n} × 8 repeats): "
          f"{10 * n * 8}\n")

    # Cargar checkpoint si se reanuda
    ckpt = None
    if args.resume and checkpoint_file:
        ckpt = _load_checkpoint(checkpoint_file)
        if ckpt is None:
            print("[INFO] No se encontró checkpoint, iniciando desde cero.")
        else:
            phase = ckpt.get("phase", "?")
            fi    = ckpt.get("current_fold_idx", 0) + 1
            zi    = ckpt.get("current_zone_idx", 0) + 1
            ep    = ckpt.get("completed_epochs", 0)
            print(f"[INFO] Reanudando desde checkpoint: fase={phase}, "
                  f"fold={fi}, zona={zi}, época completada={ep}")

    # ── LOOCV ────────────────────────────────────────────────────────────────
    all_fold_results = []
    if not args.skip_loocv:
        folds_to_run = list([0] if args.dry_run else range(n))
        print(f"[INFO] Iniciando LOOCV ({len(folds_to_run)} folds)...")

        # Parámetros de reanudación LOOCV
        start_fold_idx       = 0
        resume_zone          = 0
        resume_epoch         = 0
        fold_results_partial = None
        heads_state          = None
        opt_state            = None
        sched_state          = None

        if ckpt and ckpt.get("phase") == "loocv":
            all_fold_results     = list(ckpt.get("all_fold_results", []))
            start_fold_idx       = ckpt.get("current_fold_idx", 0)
            resume_zone          = ckpt.get("current_zone_idx", 0)
            resume_epoch         = ckpt.get("completed_epochs", 0)
            fold_results_partial = ckpt.get("fold_results_partial", {})
            heads_state          = ckpt.get("heads_state")
            opt_state            = ckpt.get("optimizer_state")
            sched_state          = ckpt.get("scheduler_state")
            print(f"[INFO] Reanudando LOOCV: fold {start_fold_idx+1}, "
                  f"zona {resume_zone+1}, época {resume_epoch+1}")

        _write_progress(args.progress_file, {
            "status":       "training",
            "phase":        "LOOCV",
            "total_folds":  len(folds_to_run),
            "total_epochs": args.epochs,
        })

        loocv_stopped = False
        for fold_idx in folds_to_run:
            if fold_idx < start_fold_idx:
                # Fold ya completado (sus resultados están en all_fold_results del checkpoint)
                continue

            fold_results, stopped = train_fold(
                fold_idx=fold_idx,
                base_samples=base_samples,
                images_dir=images_dir,
                epochs=args.epochs,
                device=device,
                models_dir=models_dir,
                dry_run=args.dry_run,
                progress_file=args.progress_file,
                total_folds=len(folds_to_run),
                checkpoint_file=checkpoint_file,
                stop_file=stop_file,
                resume_zone=resume_zone          if fold_idx == start_fold_idx else 0,
                resume_epoch=resume_epoch         if fold_idx == start_fold_idx else 0,
                resume_heads_state=heads_state    if fold_idx == start_fold_idx else None,
                resume_optimizer_state=opt_state  if fold_idx == start_fold_idx else None,
                resume_scheduler_state=sched_state if fold_idx == start_fold_idx else None,
                fold_results_partial=fold_results_partial if fold_idx == start_fold_idx else None,
                all_fold_results_so_far=all_fold_results,
            )

            if stopped:
                loocv_stopped = True
                break

            all_fold_results.append(fold_results)
            # Resetear parámetros de reanudación tras el primer fold procesado
            resume_zone          = 0
            resume_epoch         = 0
            fold_results_partial = None
            heads_state          = None
            opt_state            = None
            sched_state          = None

        if loocv_stopped:
            print("\n[PAUSED] Entrenamiento pausado. Usa 'Reanudar' en la app para continuar.")
            sys.exit(0)

        # Resumen LOOCV
        if not args.dry_run:
            print("\n" + "=" * 60)
            print("RESUMEN LOOCV")
            print("=" * 60)
            for zone_idx in range(10):
                key  = f"zona_{zone_idx}"
                ems  = [r[key]["exact_match"] for r in all_fold_results]
                maes = [r[key]["mae_per_damage"]["_global"] for r in all_fold_results]
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
        # Parámetros de reanudación modelo final
        final_start_epoch = 0
        final_heads_state = None
        final_opt_state   = None
        final_sched_state = None

        if ckpt and ckpt.get("phase") == "final":
            final_start_epoch = ckpt.get("completed_epochs", 0)
            final_heads_state = ckpt.get("heads_state")
            final_opt_state   = ckpt.get("optimizer_state")
            final_sched_state = ckpt.get("scheduler_state")
            print(f"[INFO] Reanudando modelo final desde época {final_start_epoch+1}")

        stopped = train_final_model(
            base_samples=base_samples,
            images_dir=images_dir,
            epochs=args.epochs,
            device=device,
            models_dir=models_dir,
            progress_file=args.progress_file,
            checkpoint_file=checkpoint_file,
            stop_file=stop_file,
            start_epoch=final_start_epoch,
            heads_state=final_heads_state,
            optimizer_state=final_opt_state,
            scheduler_state=final_sched_state,
            all_fold_results=all_fold_results,
        )

        if stopped:
            print("\n[PAUSED] Entrenamiento pausado. Usa 'Reanudar' en la app para continuar.")
            sys.exit(0)

        print("\n[OK] Entrenamiento completado.")
        print("     Siguiente paso: python train/export_onnx.py")
    else:
        print("\n[DRY-RUN OK] El pipeline funciona correctamente.")
        print("             Ejecuta sin --dry-run para entrenamiento completo.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[STOPPED] Entrenamiento interrumpido por el usuario.")
