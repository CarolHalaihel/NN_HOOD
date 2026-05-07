"""
export_onnx.py — Exporta HoodNet entrenado a formato ONNX para producción.

ONNX permite ejecutar el modelo sin PyTorch instalado, usando onnxruntime
que es ~3× más rápido en CPU que torch para inferencia.

Uso:
  python export_onnx.py
  python export_onnx.py --model-path ../models/hoodnet_final.pt
  python export_onnx.py --output ../models/hood_model.onnx
"""

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train.model import HoodNet


def export_hoodnet(model_path: Path, output_path: Path, verify: bool = True) -> None:
    """
    Carga HoodNet entrenado y lo exporta a ONNX con batch dinámico.

    model_path  : ruta al archivo .pt con state_dict de HoodNet
    output_path : ruta de salida para el archivo .onnx
    verify      : si True, verifica el modelo exportado con onnxruntime
    """
    print(f"[INFO] Cargando modelo desde: {model_path}")
    model = HoodNet()
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # Tensor de entrada dummy (batch=1, 3 canales, 224×224)
    dummy_input = torch.randn(1, 3, 224, 224)

    print(f"[INFO] Exportando a ONNX (opset 17): {output_path}")
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=17,
        input_names=["image"],
        output_names=["damage_logits"],
        # Batch dinámico: permite inferencia con cualquier tamaño de batch
        dynamic_axes={
            "image":          {0: "batch_size"},
            "damage_logits":  {0: "batch_size"},
        },
        do_constant_folding=True,  # optimización para CPU
    )
    print(f"[OK] Exportación completada.")

    # ── Verificación con onnxruntime ──────────────────────────────────────────
    if verify:
        try:
            import onnx
            import onnxruntime as ort
            import numpy as np

            # Verificar modelo ONNX
            onnx_model = onnx.load(str(output_path))
            onnx.checker.check_model(onnx_model)
            print("[OK] Verificación ONNX (onnx.checker) pasada.")

            # Inferencia de prueba con onnxruntime
            sess    = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
            dummy   = dummy_input.numpy()
            outputs = sess.run(None, {"image": dummy})

            logits = outputs[0]  # (1, 8, 4)
            scores = logits[0].argmax(axis=-1)  # (8,)

            print(f"[OK] Inferencia ONNX exitosa.")
            print(f"     Output shape  : {logits.shape}")
            print(f"     Scores ejemplo: {scores.tolist()}  (0=ninguno, 3=severo)")

        except ImportError as e:
            print(f"[AVISO] Verificación omitida — {e}")
            print("         Instala con: pip install onnx onnxruntime")

    # Tamaño del archivo exportado
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n[INFO] Archivo: {output_path}")
    print(f"       Tamaño : {size_mb:.1f} MB")
    print(f"\n[INFO] Siguiente paso:")
    print(f"         uvicorn api.main:app --reload")


def main():
    parser = argparse.ArgumentParser(description="Exportar HoodNet a ONNX")
    parser.add_argument(
        "--model-path", type=str, default="../models/hoodnet_final.pt",
        help="Ruta al .pt del modelo entrenado (relativo a train/)",
    )
    parser.add_argument(
        "--output", type=str, default="../models/hood_model.onnx",
        help="Ruta de salida del archivo .onnx (relativo a train/)",
    )
    parser.add_argument(
        "--no-verify", action="store_true",
        help="Omitir verificación con onnxruntime tras exportar",
    )
    args = parser.parse_args()

    train_dir   = Path(__file__).resolve().parent
    model_path  = (train_dir / args.model_path).resolve()
    output_path = (train_dir / args.output).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        print(f"[ERROR] Modelo no encontrado: {model_path}")
        print("        Ejecuta primero: python train/train.py")
        sys.exit(1)

    export_hoodnet(model_path, output_path, verify=not args.no_verify)


if __name__ == "__main__":
    main()
