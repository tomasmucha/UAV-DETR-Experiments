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
    parser = argparse.ArgumentParser(description="Build a model and run one tiny forward pass.")
    parser.add_argument("--model", required=True, help="Path to model yaml.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_arg}" if device_arg.isdigit() else device_arg)
    return torch.device("cpu")


def main():
    args = parse_args()
    if not Path(args.model).exists():
        raise FileNotFoundError(f"model yaml does not exist: {args.model}")

    device = resolve_device(args.device)
    model = RTDETR(args.model).model.to(device).eval()
    x = torch.zeros(args.batch, 3, args.imgsz, args.imgsz, device=device)
    with torch.no_grad():
        y = model(x)
    shape = y[0].shape if isinstance(y, tuple) else y.shape
    print(f"OK: built {args.model}")
    print(f"OK: forward output shape {tuple(shape)}")


if __name__ == "__main__":
    main()
