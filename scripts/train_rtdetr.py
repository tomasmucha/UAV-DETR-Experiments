import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from ultralytics import RTDETR


warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a UAV-DETR/RT-DETR experiment.")
    parser.add_argument("--model", required=True, help="Path to model yaml or checkpoint.")
    parser.add_argument("--data", required=True, help="Path to dataset yaml.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", required=True)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--resume", default=None, help="Optional last.pt path to resume from.")
    parser.add_argument("--optimizer", default="AdamW")
    parser.add_argument("--lr0", type=float, default=0.0001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.2)
    return parser.parse_args()


def require_file(path, label):
    if not Path(path).exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def main():
    args = parse_args()
    require_file(args.model, "model")
    require_file(args.data, "data yaml")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = RTDETR(args.model)
    train_kwargs = vars(args).copy()
    train_kwargs.pop("model")
    if train_kwargs["resume"] is None:
        train_kwargs.pop("resume")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
