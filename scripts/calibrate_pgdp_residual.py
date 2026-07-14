import argparse
import csv
import json
import math
import sys
import warnings
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import RTDETR
import ultralytics.nn.tasks as nn_tasks
from ultralytics.nn.uav_modules.pgdp import PGDPEnhance


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
        description="Sweep the bounded PGDP residual scale of a trained PGDP+FDR checkpoint."
    )
    parser.add_argument("--checkpoint", required=True, help="PGDP+FDR best.pt to calibrate.")
    parser.add_argument("--data", help="Dataset yaml. Required unless --inspect-only is used.")
    parser.add_argument(
        "--gammas",
        type=float,
        nargs="+",
        default=[0.0, 0.10, 0.20, 0.30, 0.36, 0.40, 0.48, 0.56, 0.68, 0.80, 1.0],
        help="Candidate effective residual scales. The checkpoint value is evaluated first.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/calibration")
    parser.add_argument("--name", default="pgdp_fdr_gamma_sweep")
    parser.add_argument("--baseline-ap", type=float, default=0.32524)
    parser.add_argument("--pgdp-ap", type=float, default=0.33368, help="Latest/final PGDP-only AP50-95.")
    parser.add_argument("--fdr-ap", type=float, default=0.33275, help="FDR-only final AP50-95.")
    parser.add_argument(
        "--interaction-fraction",
        type=float,
        default=0.5,
        help="Required fraction of the weaker single-module gain above the stronger module.",
    )
    parser.add_argument(
        "--min-sweep-gain",
        type=float,
        default=0.001,
        help="Minimum same-protocol AP50-95 gain over the original gamma to recommend retraining.",
    )
    parser.add_argument("--save-best", action="store_true", help="Save one checkpoint using the best gamma.")
    parser.add_argument("--inspect-only", action="store_true", help="Only print checkpoint PGDP gamma values.")
    return parser.parse_args()


def require_file(path, label):
    if not path or not Path(path).is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def pgdp_modules(model):
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, PGDPEnhance)]
    if not modules:
        raise RuntimeError("No PGDPEnhance module was found in the checkpoint.")
    return modules


def gamma_values(modules):
    return {name: float(module.effective_gamma.detach().cpu()) for name, module in modules}


def set_effective_gamma(modules, value):
    value = float(value)
    with torch.no_grad():
        for _, module in modules:
            gamma_max = float(module.gamma_max.detach().cpu())
            if not 0.0 <= value <= gamma_max:
                raise ValueError(f"Expected gamma in [0, {gamma_max}], got {value}")
            # Finite logits reproduce the endpoints closely without writing +/-inf.
            ratio = min(max(value / gamma_max, 1e-8), 1.0 - 1e-8)
            module.raw_gamma.fill_(math.log(ratio / (1.0 - ratio)))


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
    return {
        "gamma": gamma,
        "is_original": abs(gamma - original_gamma) <= 1e-8,
        "precision": precision,
        "recall": recall,
        "map50": map50,
        "map50_95": map50_95,
        "interaction_target": target,
        "margin_to_target": map50_95 - target,
        "passes_target": map50_95 >= target,
    }


def gamma_slug(value):
    return f"{value:.6f}".replace("-", "m").replace(".", "p")


def write_summary(rows, output_dir, metadata):
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "gamma_sweep.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "gamma_sweep.json").open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "results": rows}, handle, indent=2, ensure_ascii=False)


def save_calibrated_checkpoint(source, destination, gamma, metadata):
    checkpoint, _ = trusted_torch_safe_load(str(source))
    model = checkpoint.get("ema") or checkpoint["model"]
    set_effective_gamma(pgdp_modules(model), gamma)
    checkpoint["pgdp_residual_calibration"] = metadata
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, destination)


def main():
    args = parse_args()
    require_file(args.checkpoint, "checkpoint")

    detector = RTDETR(args.checkpoint)
    modules = pgdp_modules(detector.model)
    initial_values = gamma_values(modules)
    print("PGDP effective residual scales:", initial_values)
    if args.inspect_only:
        return
    require_file(args.data, "data yaml")

    rounded = {round(value, 8) for value in initial_values.values()}
    if len(rounded) != 1:
        raise RuntimeError(f"Expected one shared PGDP gamma value, got {initial_values}")
    original_gamma = next(iter(initial_values.values()))
    candidates = unique_values([original_gamma, *args.gammas])
    del detector
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    target = interaction_target(args.baseline_ap, args.pgdp_ap, args.fdr_ap, args.interaction_fraction)
    print(f"Dynamic interaction target AP50-95: {target:.6f}")

    rows = []
    for gamma in candidates:
        # Validation fuses Conv+BN in place, so every candidate must reload an unfused model.
        candidate_detector = RTDETR(args.checkpoint)
        set_effective_gamma(pgdp_modules(candidate_detector.model), gamma)
        run_name = f"{args.name}_gamma_{gamma_slug(gamma)}"
        metrics = candidate_detector.val(
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
        del candidate_detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    original = next(row for row in rows if row["is_original"])
    best = max(rows, key=lambda row: row["map50_95"])
    sweep_gain = best["map50_95"] - original["map50_95"]
    retrain_recommended = not best["is_original"] and sweep_gain >= args.min_sweep_gain
    output_dir = Path(args.project) / args.name
    metadata = {
        "source_checkpoint": str(Path(args.checkpoint).resolve()),
        "original_gamma": original_gamma,
        "original_map50_95": original["map50_95"],
        "best_gamma": best["gamma"],
        "best_map50_95": best["map50_95"],
        "sweep_gain": sweep_gain,
        "min_sweep_gain": args.min_sweep_gain,
        "retrain_recommended": retrain_recommended,
        "baseline_ap": args.baseline_ap,
        "pgdp_ap": args.pgdp_ap,
        "fdr_ap": args.fdr_ap,
        "interaction_fraction": args.interaction_fraction,
        "interaction_target": target,
    }
    write_summary(rows, output_dir, metadata)

    if args.save_best:
        destination = output_dir / "best_calibrated.pt"
        save_calibrated_checkpoint(args.checkpoint, destination, best["gamma"], metadata)
        print(f"Saved calibrated checkpoint: {destination}")

    print(f"Best gamma={best['gamma']:.6f}, AP50-95={best['map50_95']:.6f}")
    print(f"Same-protocol gain over original gamma: {sweep_gain:+.6f}")
    print(f"Retrain recommended: {retrain_recommended}")
    print(f"Summary: {output_dir / 'gamma_sweep.csv'}")


if __name__ == "__main__":
    main()
