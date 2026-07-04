import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import RTDETR


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate a UAV-DETR/RT-DETR checkpoint.")
    parser.add_argument("--model", required=True, help="Path to checkpoint, e.g. best.pt.")
    parser.add_argument("--data", required=True, help="Path to dataset yaml.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--save-json", action="store_true")
    return parser.parse_args()


def require_file(path, label):
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def main():
    args = parse_args()
    require_file(args.model, "model checkpoint")
    require_file(args.data, "data yaml")

    model = RTDETR(args.model)
    model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        save_json=args.save_json,
    )


if __name__ == "__main__":
    main()
