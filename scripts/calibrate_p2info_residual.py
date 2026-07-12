import argparse
import csv
import json
import sys
import warnings
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import RTDETR
import ultralytics.nn.tasks as nn_tasks
from ultralytics.nn.uav_modules.block import P2InformationEnhance


warnings.filterwarnings("ignore")


def trusted_torch_safe_load(weight):
    """Load a project-owned checkpoint across PyTorch's weights_only default change."""
    try:
        checkpoint = torch.load(weight, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch versions before weights_only was added.
        checkpoint = torch.load(weight, map_location="cpu")
    return checkpoint, str(weight)


nn_tasks.torch_safe_load = trusted_torch_safe_load


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep the P2Info residual scale of a trained P2Info+FDR checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="P2Info+FDR best.pt to calibrate.")
    parser.add_argument("--data", help="Dataset yaml. Required unless --inspect-only is used.")
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.0, 0.35, 0.45, 0.50, 0.54, 0.58, 0.62],
        help="Candidate residual scales. The checkpoint's original value is always evaluated first.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/calibration")
    parser.add_argument("--name", default="p2info_fdr_gamma_sweep")
    parser.add_argument("--baseline-ap", type=float, default=0.32524)
    parser.add_argument("--module-a-ap", type=float, default=0.33029, help="P2Info final AP50-95.")
    parser.add_argument("--module-b-ap", type=float, help="FDR-only final AP50-95.")
    parser.add_argument(
        "--interaction-fraction",
        type=float,
        default=0.5,
        help="Required fraction of the weaker single-module gain above the stronger module.",
    )
    parser.add_argument("--save-best", action="store_true", help="Save one checkpoint using the best swept gamma.")
    parser.add_argument("--inspect-only", action="store_true", help="Only print checkpoint P2Info gamma values.")
    return parser.parse_args()


def require_file(path, label):
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def p2info_modules(model):
    modules = []
    for name, module in model.named_modules():
        if isinstance(module, P2InformationEnhance) and hasattr(module, "gamma"):
            modules.append((name, module))
    if not modules:
        raise RuntimeError("No original P2InformationEnhance module with a gamma parameter was found.")
    return modules


def gamma_values(modules):
    return {name: float(module.gamma.detach().cpu()) for name, module in modules}


def set_gamma(modules, value):
    with torch.no_grad():
        for _, module in modules:
            module.gamma.fill_(float(value))


def unique_values(values, tolerance=1e-8):
    result = []
    for value in values:
        value = float(value)
        if not any(abs(value - existing) <= tolerance for existing in result):
            result.append(value)
    return result


def interaction_target(baseline, module_a, module_b, fraction):
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"interaction-fraction must be in [0, 1], got {fraction}")
    gain_a = max(float(module_a) - float(baseline), 0.0)
    gain_b = max(float(module_b) - float(baseline), 0.0)
    return max(float(module_a), float(module_b)) + fraction * min(gain_a, gain_b)


def metric_row(metrics, gamma, original_gamma, target):
    precision, recall, map50, map50_95 = (float(x) for x in metrics.mean_results())
    row = {
        "gamma": gamma,
        "is_original": abs(gamma - original_gamma) <= 1e-8,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
    }
    if target is not None:
        row["interaction_target"] = target
        row["margin_to_target"] = map50_95 - target
        row["passes_target"] = map50_95 >= target
    return row


def gamma_slug(value):
    return f"{value:.4f}".replace("-", "m").replace(".", "p")


def write_summary(rows, output_dir, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with (output_dir / "gamma_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "gamma_sweep.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "results": rows}, handle, indent=2, ensure_ascii=False)


def save_calibrated_checkpoint(source, destination, gamma, metadata):
    checkpoint, _ = trusted_torch_safe_load(str(source))
    model = checkpoint.get("ema") or checkpoint["model"]
    modules = p2info_modules(model)
    set_gamma(modules, gamma)
    checkpoint["p2info_residual_calibration"] = metadata
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def main():
    args = parse_args()
    require_file(args.checkpoint, "checkpoint")

    detector = RTDETR(args.checkpoint)
    modules = p2info_modules(detector.model)
    initial_values = gamma_values(modules)
    print("P2Info residual scales:", initial_values)

    if args.inspect_only:
        return
    require_file(args.data, "data yaml")

    rounded = {round(value, 8) for value in initial_values.values()}
    if len(rounded) != 1:
        raise RuntimeError(f"Expected one shared P2Info gamma value, got {initial_values}")
    original_gamma = next(iter(initial_values.values()))
    candidates = unique_values([original_gamma, *args.gammas])

    target = None
    if args.module_b_ap is not None:
        target = interaction_target(
            args.baseline_ap,
            args.module_a_ap,
            args.module_b_ap,
            args.interaction_fraction,
        )
        print(f"Dynamic interaction target AP50-95: {target:.5f}")

    rows = []
    for gamma in candidates:
        set_gamma(modules, gamma)
        run_name = f"{args.name}_gamma_{gamma_slug(gamma)}"
        metrics = detector.val(
            data=args.data,
            split="val",
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            project=args.project,
            name=run_name,
            plots=False,
            save_json=False,
            exist_ok=True,
        )
        row = metric_row(metrics, gamma, original_gamma, target)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    best = max(rows, key=lambda row: row["map50_95"])
    output_dir = Path(args.project) / args.name
    metadata = {
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "original_gamma": original_gamma,
        "best_gamma": best["gamma"],
        "best_map50_95": best["map50_95"],
        "baseline_ap": args.baseline_ap,
        "module_a_ap": args.module_a_ap,
        "module_b_ap": args.module_b_ap,
        "interaction_fraction": args.interaction_fraction,
        "interaction_target": target,
    }
    write_summary(rows, output_dir, metadata)

    if args.save_best:
        destination = output_dir / "best_calibrated.pt"
        save_calibrated_checkpoint(args.checkpoint, destination, best["gamma"], metadata)
        print(f"Saved calibrated checkpoint: {destination}")

    print(f"Best gamma={best['gamma']:.6f}, AP50-95={best['map50_95']:.6f}")
    print(f"Summary: {output_dir / 'gamma_sweep.csv'}")


if __name__ == "__main__":
    main()
