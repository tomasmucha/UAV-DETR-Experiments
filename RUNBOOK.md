# UAV-DETR Experiment Runbook

This repository keeps code and configs only. Do not commit datasets, checkpoints, or `runs/` outputs.

## Local Layout

Repository root:

```text
D:\Projects\UAV-DETR-main
```

Local dataset:

```text
D:\Projects\UAV-DETR-main\datasets\VisDrone2019-DET
```

Local experiment archive:

```text
D:\Projects\UAV-DETR-main\results_archive
```

Failed or low-priority experiments:

```text
D:\Projects\UAV-DETR-main\results_archive\graveyard
```

## Current Baselines

Best known baseline:

```text
run: uavdetr_r18_baseline_visdrone640
config: ultralytics/cfg/models/uavdetr-r18.yaml
best epoch: 258
AP50: 0.52308
AP50-95: 0.32524
P: 0.64197
R: 0.50290
```

Best known improved run:

```text
run: uavdetr_r18_nrp3_visdrone640
config: ultralytics/cfg/models/uavdetr-r18-nrp3.yaml
best epoch: 271
AP50: 0.52638
AP50-95: 0.32886
P: 0.63733
R: 0.50875
```

Highest observed P2Info checkpoint:

```text
run: uavdetr_r18_p2info_visdrone640
config: ultralytics/cfg/models/uavdetr-r18-p2info.yaml
early-stop epoch: 256 (`patience=20`)
best epoch: 236
AP50: 0.52649
AP50-95: 0.33029
P: 0.64205
R: 0.50786
path: results_archive/uavdetr_r18_p2info_visdrone640
```

Treat P2Info as the current highest observed checkpoint and NRP3 as the established completed A-module reference. P2Info improves AP50-95 by 0.00505 over baseline and 0.00143 over NRP3.

The 400-epoch setting is a maximum budget. Use the shared `patience=20` early-stopping rule and report each run's best checkpoint; do not spend GPU time forcing a converged run to epoch 400.

Completed P2Info + NRP3 combination ablation:

```text
run: uavdetr_r18_p2info_nrp3_visdrone640
config: ultralytics/cfg/models/uavdetr-r18-p2info-nrp3.yaml
early-stop epoch: 258 (`patience=20`)
best epoch: 238
AP50: 0.51951
AP50-95: 0.32631
P: 0.63863
R: 0.49928
path: results_archive/uavdetr_r18_p2info_nrp3_visdrone640
```

Keep this run in the main archive as a formal A+B ablation. It exceeds the baseline AP50-95 by 0.00107 but remains below NRP3 by 0.00255 and P2Info by 0.00398, so it is a no-synergy result rather than the final combined method.

## AutoDL Training Template

```bash
cd /root/autodl-tmp/UAV-DETR-Experiments
/root/miniconda3/bin/python scripts/train_rtdetr.py \
  --model ultralytics/cfg/models/uavdetr-r18-nrp3.yaml \
  --data /root/autodl-tmp/datasets/VisDrone/visdrone.yaml \
  --imgsz 640 \
  --epochs 400 \
  --batch 4 \
  --workers 8 \
  --device 0 \
  --project runs/train \
  --name uavdetr_r18_nrp3_visdrone640
```

Use a distinct `--name` for every experiment. Never hardcode dataset paths into model code.

## Smoke Test

```bash
cd /root/autodl-tmp/UAV-DETR-Experiments
/root/miniconda3/bin/python -m compileall -q ultralytics scripts train.py val.py
/root/miniconda3/bin/python scripts/smoke_model.py \
  --model ultralytics/cfg/models/uavdetr-r18-nrp3.yaml \
  --imgsz 640 \
  --device 0
```

## tmux Training

```bash
tmux new -d -s uavdetr_exp 'bash logs/run_full_train.sh'
tmux attach -t uavdetr_exp
```

Detach without stopping:

```text
Ctrl+B, then D
```

Stop only after deciding the run should end:

```bash
tmux send-keys -t uavdetr_exp C-c
```

## Validation

If a run is interrupted manually, run validation from `best.pt` to generate PR curves, F1 curves, confusion matrices, and prediction images.

```bash
cd /root/autodl-tmp/UAV-DETR-Experiments
/root/miniconda3/bin/python scripts/val_rtdetr.py \
  --model runs/train/<run_name>/weights/best.pt \
  --data /root/autodl-tmp/datasets/VisDrone/visdrone.yaml \
  --imgsz 640 \
  --batch 4 \
  --device 0 \
  --project runs/val \
  --name <run_name>_best \
  --save-json
```

## Early Stop Rule

Compare every new run against baseline and A v1 at the same epoch. Do not judge only by the final best.

Useful checkpoint epochs:

```text
20, 30, 50, 75, 100, 135, 160, 180, 200, 250
```

Stop early when:

- the run is below baseline and A v1 for many checkpoints,
- recall, precision, AP50, and AP50-95 show no compensating advantage,
- best AP has not refreshed for about 10 epochs after epoch 200,
- or the run matches a known failed direction in `EXPERIMENT_NOTES.md`.

## Low-Priority Directions

Do not prioritize these unless they are needed for an ablation table:

```text
MSNoiseGate / noisegate
NRP3 + noisegate
P2-guided P3 direct injection
hard SBQ / scale-balanced query selection
Soft-SBQ
NRP3 + Soft-SBQ
decoder-input cross-scale adapter
ContextRepC3 / NeckCtx
```
