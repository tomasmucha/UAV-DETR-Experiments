# Experiment Notes

## Baseline: UAV-DETR-R18

The reproduced baseline uses `ultralytics/cfg/models/uavdetr-r18.yaml` on VisDrone at `imgsz=640`.

Observed validation behavior from `runs/train/uavdetr_r18_visdrone_640`:

- Best mAP50-95 appears around epoch 258: mAP50 about 0.523, mAP50-95 about 0.325.
- Recall plateaus around 0.50, while precision is higher at about 0.64.
- The normalized confusion matrix shows a clear background column/row issue: many true objects are missed as background.
- The miss pattern is strongest for small or visually ambiguous classes such as car, pedestrian/person, motorbike, bicycle, and tricycle.
- There is also inter-class confusion among visually similar vehicle classes, especially bus/van/truck and motorbike/bicycle.

## First Ablation: P3Refine

File: `ultralytics/cfg/models/uavdetr-r18-p3.yaml`

The baseline already feeds P3/P4/P5 into `RTDETRDecoder`. Therefore this ablation does not add a new detection scale. It keeps the architecture change localized by refining only the P3 feature immediately before the decoder.

The module `P3Refine` is intentionally small:

- 1x1 channel reduction.
- Depthwise 3x3 local detail branch.
- Dilated depthwise 5x5 context branch.
- Lightweight channel gate.
- 1x1 projection back to the original P3 channel size.
- Residual output with a learnable scale.

Expected effect:

- Improve small-object detail before query decoding.
- Add limited context to suppress noisy background responses.
- Keep parameter and FLOP growth small enough for the UAV/edge deployment motivation.

Main risk:

- P3 is high resolution, so even small modules add some cost.
- If the gate over-suppresses weak object features, recall may not improve.
