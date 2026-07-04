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

## Main Proposed Direction: Noise-Robust Small-Object Enhancement

The stronger experiment family targets the current failure pattern more directly:

- Small and dense objects are frequently missed as background.
- Similar traffic categories are confused with each other.
- UAV images include cluttered backgrounds, compression artifacts, blur, and low-quality regions.
- The original UAV-DETR paper also states that robustness to noisy UAV inputs is a future direction.

### Module 1: NRP3CBAM

File: `ultralytics/nn/uav_modules/block.py`

Config: `ultralytics/cfg/models/uavdetr-r18-nrp3.yaml`

`NRP3CBAM` refines only the P3 high-resolution feature before the RT-DETR decoder. It combines:

- Multi-scale depthwise convolution for local detail and wider context.
- Existing frequency-style enhancement through `FFM`.
- CBAM-style channel attention and spatial attention.
- Residual scaling to reduce optimization risk.

Expected effect:

- Recover weak small-object cues before query decoding.
- Reduce background distraction around dense targets.
- Improve recall without completely changing the UAV-DETR neck.

### Module 2: MSNoiseGate

File: `ultralytics/nn/uav_modules/block.py`

Config: `ultralytics/cfg/models/uavdetr-r18-noisegate.yaml`

`MSNoiseGate` is applied separately to P3/P4/P5 before the decoder. It estimates local high-frequency residuals as a simple noise cue and uses a learnable gate to balance the original feature, a smoothed feature, and local structure.

Expected effect:

- Suppress noisy background responses before decoder query selection.
- Improve feature stability across P3/P4/P5.
- Support the paper-level theme of improving UAV-DETR robustness to noisy inputs.

### Combined Model

Config: `ultralytics/cfg/models/uavdetr-r18-nrp3-noisegate.yaml`

This is the recommended first paid GPU experiment:

- `NRP3CBAM` strengthens P3 for small objects.
- `MSNoiseGate` filters P3/P4/P5 before the decoder.

If the combined model improves over baseline, then run the two single-module configs for ablation:

1. `uavdetr-r18-nrp3.yaml`
2. `uavdetr-r18-noisegate.yaml`
3. `uavdetr-r18-nrp3-noisegate.yaml`

This saves training cost because the full combination is tested first.
