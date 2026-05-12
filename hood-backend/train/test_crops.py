"""Smoke test for ZoneCropDataset pipeline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.dataset import ZoneCropDataset, load_base_samples

images_dir       = Path("data/images")
annotations_path = Path("data/annotations.json")

base = load_base_samples(images_dir, annotations_path, only_aE=True)
print(f"Base samples: {len(base)} imagenes")
for name, ann in base:
    n_lm = len(ann.get("landmarks", {}))
    print(f"  {name}  landmarks={n_lm}")

if not base:
    print("ERROR: no base samples found")
    sys.exit(1)

ds = ZoneCropDataset(
    images_dir=str(images_dir),
    base_samples=base,
    zone_idx=0,
    augment=True,
    image_size=224,
    repeat_factor=2,
)
print(f"\nZoneCropDataset zona 0: {len(ds)} muestras (repeat_factor=2)")
sample = ds[0]
print(f"  image shape    : {sample['image'].shape}")
print(f"  damage_scores  : {sample['damage_scores'].tolist()}")
print(f"  zone_idx       : {sample['zone_idx']}")
print(f"  image_name     : {sample['image_name']}")
print("\nZoneCropDataset OK")

# Test train dry run imports
print("\nVerificando importaciones de train.py...")
from train.train import train_one_epoch, evaluate_zone, train_fold, load_base_samples as _lbs
print("train.py imports OK")
print("\nTodo OK - pipeline de recortes listo.")
