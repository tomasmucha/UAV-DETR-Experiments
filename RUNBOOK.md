# UAV-DETR Experiment Runbook

This repository keeps code and configs only. Do not commit datasets, checkpoints, or `runs/` outputs.

## Baseline Training on AutoDL

```bash
cd /root/autodl-tmp/UAV-DETR-Experiments
python scripts/train_rtdetr.py \
  --model ultralytics/cfg/models/uavdetr-r18.yaml \
  --data /root/autodl-tmp/datasets/VisDrone/visdrone.yaml \
  --imgsz 640 \
  --epochs 400 \
  --batch 4 \
  --workers 8 \
  --device 0 \
  --project runs/train \
  --name uavdetr_r18_baseline_visdrone640
```

## P3 Experiment Training on AutoDL

```bash
cd /root/autodl-tmp/UAV-DETR-Experiments
python scripts/train_rtdetr.py \
  --model ultralytics/cfg/models/uavdetr-r18-p3.yaml \
  --data /root/autodl-tmp/datasets/VisDrone/visdrone.yaml \
  --imgsz 640 \
  --epochs 400 \
  --batch 4 \
  --workers 8 \
  --device 0 \
  --project runs/train \
  --name uavdetr_r18_p3_visdrone640
```

## Smoke Test

```bash
python -m compileall -q ultralytics scripts train.py val.py
python scripts/smoke_model.py --model ultralytics/cfg/models/uavdetr-r18.yaml --imgsz 640 --device cpu
python scripts/smoke_model.py --model ultralytics/cfg/models/uavdetr-r18-p3.yaml --imgsz 640 --device cpu
```

## Validation

```bash
python scripts/val_rtdetr.py \
  --model runs/train/uavdetr_r18_p3_visdrone640/weights/best.pt \
  --data /root/autodl-tmp/datasets/VisDrone/visdrone.yaml \
  --imgsz 640 \
  --batch 4 \
  --device 0 \
  --project runs/val \
  --name uavdetr_r18_p3_visdrone640
```
