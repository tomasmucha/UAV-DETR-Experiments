# UAV-DETR Experiment Runbook

This repository keeps code and configs only. Do not commit datasets, checkpoints, or `runs/` outputs.

## Baseline Training on AutoDL

```bash
cd /root/autodl-tmp/UAV-DETR-Experiments
/root/miniconda3/bin/python scripts/train_rtdetr.py \
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
/root/miniconda3/bin/python scripts/train_rtdetr.py \
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

## Recommended Strong Experiment on AutoDL

Run this after smoke testing. It uses high-resolution P2 detail to guide the final P3 feature before the RT-DETR decoder.

```bash
cd /root/autodl-tmp/UAV-DETR-Experiments
/root/miniconda3/bin/python scripts/train_rtdetr.py \
  --model ultralytics/cfg/models/uavdetr-r18-p2p3.yaml \
  --data /root/autodl-tmp/datasets/VisDrone/visdrone.yaml \
  --imgsz 640 \
  --epochs 400 \
  --batch 4 \
  --workers 8 \
  --device 0 \
  --project runs/train \
  --name uavdetr_r18_p2p3_visdrone640
```

Previous ablations kept for comparison:

```bash
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

/root/miniconda3/bin/python scripts/train_rtdetr.py \
  --model ultralytics/cfg/models/uavdetr-r18-noisegate.yaml \
  --data /root/autodl-tmp/datasets/VisDrone/visdrone.yaml \
  --imgsz 640 \
  --epochs 400 \
  --batch 4 \
  --workers 8 \
  --device 0 \
  --project runs/train \
  --name uavdetr_r18_noisegate_visdrone640
```

## Smoke Test

```bash
/root/miniconda3/bin/python -m compileall -q ultralytics scripts train.py val.py
/root/miniconda3/bin/python scripts/smoke_model.py --model ultralytics/cfg/models/uavdetr-r18.yaml --imgsz 640 --device 0
/root/miniconda3/bin/python scripts/smoke_model.py --model ultralytics/cfg/models/uavdetr-r18-p2p3.yaml --imgsz 640 --device 0
/root/miniconda3/bin/python scripts/train_rtdetr.py \
  --model ultralytics/cfg/models/uavdetr-r18-p2p3.yaml \
  --data /root/autodl-tmp/datasets/VisDrone/visdrone.yaml \
  --imgsz 640 \
  --epochs 1 \
  --batch 4 \
  --workers 8 \
  --device 0 \
  --project runs/train \
  --name smoke_r18_p2p3_fast \
  --exist-ok
```

## Validation

```bash
/root/miniconda3/bin/python scripts/val_rtdetr.py \
  --model runs/train/uavdetr_r18_p3_visdrone640/weights/best.pt \
  --data /root/autodl-tmp/datasets/VisDrone/visdrone.yaml \
  --imgsz 640 \
  --batch 4 \
  --device 0 \
  --project runs/val \
  --name uavdetr_r18_p3_visdrone640
```
