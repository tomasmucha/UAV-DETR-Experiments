"""Audit PGDP/FDR loss interactions on archived checkpoints.

The script runs real VisDrone training batches without optimizer steps and
measures gradient norms and pairwise cosine similarities for PGDP auxiliary,
classification, box/IoU and FGL losses on representative shared layers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import RTDETR
from ultralytics.cfg import get_cfg
from ultralytics.data import build_dataloader
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.rtdetr.val import RTDETRDataset
from ultralytics.utils.torch_utils import select_device


# Archived checkpoints are full, locally-produced model objects rather than
# untrusted weights-only files. PyTorch 2.6 changed torch.load's default.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")


LOSS_GROUPS = ("pgdp", "classification", "box", "fgl")
LOSS_PAIRS = tuple((a, b) for i, a in enumerate(LOSS_GROUPS) for b in LOSS_GROUPS[i + 1 :])


def parse_args():
    parser = argparse.ArgumentParser(description="Measure PGDP/FDR shared-gradient interactions.")
    parser.add_argument("--checkpoint", nargs="+", required=True, help="Archived PGDP+FDR best.pt/last.pt files.")
    parser.add_argument("--data", required=True, help="VisDrone data yaml.")
    parser.add_argument("--batches", type=int, default=32, help="Number of real training batches per checkpoint.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--mixup", type=float, default=0.2)
    parser.add_argument(
        "--layer-prefixes",
        default="model.4,model.5,model.6,model.21,model.28",
        help="Comma-separated shared-layer parameter prefixes.",
    )
    parser.add_argument("--output-dir", default="runs/audit/pgdp_fdr_gradient_audit")
    return parser.parse_args()


def build_loader(args):
    data = check_det_dataset(args.data)
    hyp = get_cfg(
        overrides={
            "task": "detect",
            "mode": "train",
            "imgsz": args.imgsz,
            "batch": args.batch,
            "workers": args.workers,
            "mosaic": args.mosaic,
            "mixup": args.mixup,
            "device": args.device,
            "seed": 0,
            "deterministic": True,
            "amp": False,
        }
    )
    dataset = RTDETRDataset(
        img_path=data["train"],
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=True,
        hyp=hyp,
        rect=False,
        cache=None,
        prefix="audit: ",
        fraction=1.0,
        data=data,
    )
    return build_dataloader(dataset, args.batch, args.workers, shuffle=True, rank=-1)


def make_targets(batch, device):
    image = batch["img"].to(device, non_blocking=True).float() / 255.0
    batch_idx = batch["batch_idx"].to(device=device, dtype=torch.long).view(-1)
    return image, {
        "cls": batch["cls"].to(device=device, dtype=torch.long).view(-1),
        "bboxes": batch["bboxes"].to(device=device),
        "batch_idx": batch_idx,
        "gt_groups": [(batch_idx == i).sum().item() for i in range(len(image))],
    }


def compute_loss_dict(model, image, targets):
    """Mirror RTDETRDetectionModel.loss while retaining the component dictionary."""
    criterion = model.criterion
    aux_modules = [
        module
        for module in model.modules()
        if hasattr(module, "set_targets") and hasattr(module, "consume_aux_loss")
    ]
    for module in aux_modules:
        module.set_targets(targets, image.shape[-2:])

    outputs = model.predict(image, batch=targets)
    dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = outputs[:5]
    fdr_outputs = outputs[5:] if len(outputs) > 5 else None

    dn_bboxes = dn_scores = dn_corners = dn_refs = None
    fdr_dn_bboxes = fdr_dn_scores = None
    if dn_meta is None:
        if fdr_outputs:
            dec_corners, dec_refs, pre_bboxes, pre_scores = fdr_outputs
            fdr_outputs = dec_corners, dec_refs
    else:
        dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta["dn_num_split"], dim=2)
        dn_scores, dec_scores = torch.split(dec_scores, dn_meta["dn_num_split"], dim=2)
        if fdr_outputs:
            dec_corners, dec_refs, pre_bboxes, pre_scores = fdr_outputs
            fdr_dn_bboxes, fdr_dn_scores = dn_bboxes, dn_scores
            dn_corners, dec_corners = torch.split(dec_corners, dn_meta["dn_num_split"], dim=2)
            dn_refs, dec_refs = torch.split(dec_refs, dn_meta["dn_num_split"], dim=2)
            dn_pre_bboxes, pre_bboxes = torch.split(pre_bboxes, dn_meta["dn_num_split"], dim=1)
            dn_pre_scores, pre_scores = torch.split(pre_scores, dn_meta["dn_num_split"], dim=1)
            dn_bboxes = torch.cat((dn_pre_bboxes.unsqueeze(0), dn_bboxes))
            dn_scores = torch.cat((dn_pre_scores.unsqueeze(0), dn_scores))
            fdr_outputs = dec_corners, dec_refs

    fdr_dec_bboxes, fdr_dec_scores = dec_bboxes, dec_scores
    if fdr_outputs:
        dec_bboxes = torch.cat((enc_bboxes.unsqueeze(0), pre_bboxes.unsqueeze(0), dec_bboxes))
        dec_scores = torch.cat((enc_scores.unsqueeze(0), pre_scores.unsqueeze(0), dec_scores))
    else:
        dec_bboxes = torch.cat((enc_bboxes.unsqueeze(0), dec_bboxes))
        dec_scores = torch.cat((enc_scores.unsqueeze(0), dec_scores))

    losses = criterion(
        (dec_bboxes, dec_scores),
        targets,
        dn_bboxes=dn_bboxes,
        dn_scores=dn_scores,
        dn_meta=dn_meta,
    )
    if fdr_outputs:
        dec_corners, dec_refs = fdr_outputs
        losses.update(
            criterion.forward_fdr(
                fdr_dec_bboxes,
                fdr_dec_scores,
                dec_corners,
                dec_refs,
                targets,
                dn_bboxes=fdr_dn_bboxes,
                dn_scores=fdr_dn_scores,
                dn_corners=dn_corners,
                dn_refs=dn_refs,
                dn_meta=dn_meta,
                pre_bboxes=pre_bboxes,
                pre_scores=pre_scores,
                enc_bboxes=enc_bboxes,
                enc_scores=enc_scores,
            )
        )
    for index, module in enumerate(aux_modules):
        feature_loss = module.consume_aux_loss()
        if feature_loss is not None:
            losses[f"loss_feature_aux_{index}"] = feature_loss
    return losses


def group_losses(losses):
    first = next(iter(losses.values()))
    groups = {name: first.new_zeros(()) for name in LOSS_GROUPS}
    for key, value in losses.items():
        if key.startswith("loss_feature_aux"):
            groups["pgdp"] = groups["pgdp"] + value
        elif key.startswith("loss_class"):
            groups["classification"] = groups["classification"] + value
        elif key.startswith("loss_bbox") or key.startswith("loss_giou"):
            groups["box"] = groups["box"] + value
        elif key.startswith("loss_fgl"):
            groups["fgl"] = groups["fgl"] + value
    missing = [name for name, value in groups.items() if not value.requires_grad]
    if missing:
        raise RuntimeError(f"Loss groups without gradients: {missing}; keys={sorted(losses)}")
    return groups


def select_parameters(model, prefixes):
    selected = []
    scopes = {prefix: [] for prefix in prefixes}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        for prefix in prefixes:
            if name == prefix or name.startswith(prefix + "."):
                scopes[prefix].append(len(selected))
                selected.append((name, parameter))
                break
    missing = [prefix for prefix, indices in scopes.items() if not indices]
    if missing:
        preview = [name for name, _ in list(model.named_parameters())[:30]]
        raise RuntimeError(f"No parameters matched prefixes {missing}. Parameter preview: {preview}")
    scopes = {"all_shared": list(range(len(selected))), **scopes}
    return selected, scopes


def gradient_stats(groups, parameters, scopes):
    params = [parameter for _, parameter in parameters]
    gradients = {}
    for index, name in enumerate(LOSS_GROUPS):
        gradients[name] = torch.autograd.grad(
            groups[name],
            params,
            retain_graph=index < len(LOSS_GROUPS) - 1,
            allow_unused=True,
        )

    rows = {}
    for scope, indices in scopes.items():
        row = {}
        norms = {}
        for name in LOSS_GROUPS:
            norm_sq = groups[name].new_zeros((), dtype=torch.float32)
            for idx in indices:
                grad = gradients[name][idx]
                if grad is not None:
                    norm_sq = norm_sq + grad.detach().float().square().sum()
            norms[name] = norm_sq.sqrt()
            row[f"norm_{name}"] = float(norms[name].cpu())

        for first, second in LOSS_PAIRS:
            dot = groups[first].new_zeros((), dtype=torch.float32)
            for idx in indices:
                grad_first, grad_second = gradients[first][idx], gradients[second][idx]
                if grad_first is not None and grad_second is not None:
                    dot = dot + (grad_first.detach().float() * grad_second.detach().float()).sum()
            denom = norms[first] * norms[second]
            cosine = dot / denom if float(denom) > 0 else dot.new_tensor(float("nan"))
            row[f"cos_{first}_{second}"] = float(cosine.cpu())
        rows[scope] = row
    return rows


def summarize(rows):
    summaries = []
    keys = [key for key in rows[0] if key.startswith("norm_") or key.startswith("cos_")]
    for checkpoint in sorted({row["checkpoint"] for row in rows}):
        for scope in sorted({row["scope"] for row in rows if row["checkpoint"] == checkpoint}):
            subset = [row for row in rows if row["checkpoint"] == checkpoint and row["scope"] == scope]
            summary = {"checkpoint": checkpoint, "scope": scope, "batches": len(subset)}
            for key in keys:
                values = [float(row[key]) for row in subset if math.isfinite(float(row[key]))]
                summary[f"mean_{key}"] = statistics.fmean(values) if values else float("nan")
                summary[f"median_{key}"] = statistics.median(values) if values else float("nan")
                if key.startswith("cos_"):
                    summary[f"negative_fraction_{key}"] = (
                        sum(value < 0 for value in values) / len(values) if values else float("nan")
                    )
            summaries.append(summary)
    return summaries


def main():
    args = parse_args()
    if args.batches <= 0:
        raise ValueError("--batches must be positive")
    checkpoints = [Path(path).resolve() for path in args.checkpoint]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device, batch=args.batch)
    prefixes = [item.strip() for item in args.layer_prefixes.split(",") if item.strip()]
    loader = build_loader(args)
    rows = []

    for checkpoint in checkpoints:
        detector = RTDETR(str(checkpoint))
        model = detector.model.to(device)
        # Ultralytics checkpoints prefer EMA weights whose parameters are
        # frozen for evaluation; the audit needs gradients but performs no step.
        model.requires_grad_(True)
        model.train()
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
        model.criterion = model.init_criterion()
        parameters, scopes = select_parameters(model, prefixes)
        scope_metadata = {
            scope: {
                "tensor_count": len(indices),
                "parameter_count": sum(parameters[index][1].numel() for index in indices),
            }
            for scope, indices in scopes.items()
        }
        print(f"Auditing {checkpoint.name}: {scope_metadata}")

        for batch_index, batch in enumerate(loader):
            if batch_index >= args.batches:
                break
            image, targets = make_targets(batch, device)
            losses = compute_loss_dict(model, image, targets)
            groups = group_losses(losses)
            stats = gradient_stats(groups, parameters, scopes)
            loss_values = {f"loss_{name}": float(value.detach().cpu()) for name, value in groups.items()}
            for scope, scope_stats in stats.items():
                rows.append(
                    {
                        "checkpoint": checkpoint.name,
                        "batch_index": batch_index,
                        "scope": scope,
                        **loss_values,
                        **scope_stats,
                    }
                )
            model.zero_grad(set_to_none=True)
            print(
                f"{checkpoint.name} batch {batch_index + 1}/{args.batches}: "
                f"cos(pgdp,fgl)={stats['all_shared']['cos_pgdp_fgl']:.4f}, "
                f"cos(pgdp,box)={stats['all_shared']['cos_pgdp_box']:.4f}"
            )

        del model, detector
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("No audit rows were produced")
    summaries = summarize(rows)
    csv_path = output_dir / "gradient_audit_batches.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "gradient_audit_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "checkpoints": [str(path) for path in checkpoints],
                "batches": args.batches,
                "batch_size": args.batch,
                "imgsz": args.imgsz,
                "layer_prefixes": prefixes,
                "summaries": summaries,
            },
            indent=2,
            allow_nan=True,
        ),
        encoding="utf-8",
    )
    print(f"Saved {csv_path}")
    print(f"Saved {json_path}")
    for summary in summaries:
        if summary["scope"] == "all_shared":
            print(json.dumps(summary, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
