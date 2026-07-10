# UAV-DETR Experiment Notes

## Current Context

The project studies UAV-DETR-R18 on VisDrone2019-DET for small, dense, and occluded object detection. The main baseline issue is missed detections: many true objects are classified as background. This is especially visible for small or ambiguous classes such as pedestrian, person, car, motorbike, bicycle, and tricycle.

The original paper mentions noise robustness as future work, but the experiments so far show that simply smoothing or gating noisy details can hurt small-object detection.

## Confirmed Reference Results

Baseline:

```text
config: ultralytics/cfg/models/uavdetr-r18.yaml
run: uavdetr_r18_baseline_visdrone640
best epoch: 258
P: 0.64197
R: 0.50290
AP50: 0.52308
AP50-95: 0.32524
```

Current best ablation, A v1:

```text
config: ultralytics/cfg/models/uavdetr-r18-nrp3.yaml
run: uavdetr_r18_nrp3_visdrone640
best epoch: 271
P: 0.63733
R: 0.50875
AP50: 0.52638
AP50-95: 0.32886
```

P2Info, stopped manually at epoch 256:

```text
config: ultralytics/cfg/models/uavdetr-r18-p2info.yaml
run: uavdetr_r18_p2info_visdrone640
best epoch: 236
P: 0.64205
R: 0.50786
AP50: 0.52649
AP50-95: 0.33029
path: results_archive/uavdetr_r18_p2info_visdrone640
```

P2Info is the highest observed checkpoint so far. It exceeds the baseline best by 0.00505 AP50-95 and NRP3 best by 0.00143. The last epoch remained at 0.33016 AP50-95, so the gain is not a single-epoch spike.

## What Has Worked

### P2Info: Information-Guided P2 Enhancement

File:

```text
ultralytics/cfg/models/uavdetr-r18-p2info.yaml
```

Interpretation:

- Selective P2 enhancement is independently effective and stronger than generic P2-to-P3 detail injection.
- The final 30 epochs all exceeded the baseline final best; 19 of 30 exceeded the NRP3 final best.
- Keep P2Info as the strongest current B module and use its archived best checkpoint for final validation and ablation.

### A v1: NRP3CBAM

File:

```text
ultralytics/cfg/models/uavdetr-r18-nrp3.yaml
```

Idea:

- refine final P3 before the decoder,
- use multi-scale depthwise context,
- use frequency-style enhancement through existing `FFM`,
- use CBAM-style channel and spatial attention,
- keep residual scaling to reduce optimization risk.

Result:

```text
AP50: +0.00330 over baseline
AP50-95: +0.00362 over baseline
Recall: +0.00585 over baseline
Precision: -0.00464 below baseline
```

Interpretation:

- The high-resolution P3 direction is reasonable.
- The effect is too weak, so future work needs a stronger but more selective module.

## Failed Or Low-Priority Directions

### MSNoiseGate

Config:

```text
ultralytics/cfg/models/uavdetr-r18-noisegate.yaml
```

Best result:

```text
best epoch: 243
P: 0.63099
R: 0.49657
AP50: 0.51047
AP50-95: 0.31056
```

Why it failed:

- It likely suppresses high-frequency small-object details together with noise.
- Recall and AP both dropped.
- Noise robustness cannot be implemented as generic smoothing for this task.

### NRP3 + NoiseGate

Config:

```text
ultralytics/cfg/models/uavdetr-r18-nrp3-noisegate.yaml
```

Best result:

```text
best epoch: 169
P: 0.63258
R: 0.48842
AP50: 0.50599
AP50-95: 0.31367
```

Why it failed:

- NRP3 tries to recover fine detail, while NoiseGate suppresses high-frequency responses.
- The two modules interact negatively.
- This confirms that recall-oriented P3 enhancement and noise smoothing should not be naively combined.

### A v2: P2-Guided P3

Config:

```text
ultralytics/cfg/models/uavdetr-r18-p2p3.yaml
```

Best result:

```text
best epoch: 250
P: 0.63279
R: 0.50824
AP50: 0.52093
AP50-95: 0.32316
```

Why it failed:

- Directly injecting P2 detail into P3 increased recall slightly but reduced precision and AP.
- P2 contains useful small-object detail, but also more background texture and noise.
- The module was more aggressive than A v1, but not selective enough.

### Hard SBQ / Scale-Balanced Query Selection

Run:

```text
uavdetr_r18_sbq_visdrone640
```

Observed result:

```text
best around epoch 85
P: 0.59783
R: 0.44824
AP50: 0.46598
AP50-95: 0.28916
```

Why it failed:

- Reserving more P3 queries does not improve the quality of P3 features.
- If P3 candidates are weak, quota allocation only preserves more weak candidates.
- The bottleneck is feature quality and class-background separability, not only query count.

### NRP3 + Soft-SBQ

Run:

```text
uavdetr_r18_nrp3_softsbq_visdrone640
```

Observed result:

```text
stopped at epoch 21
P: 0.45547
R: 0.31894
AP50: 0.32087
AP50-95: 0.18866
```

Why it failed:

- Soft query reservation did not cooperate with NRP3.
- It likely disrupted the decoder's original confidence ranking.
- Query allocation is not the current priority.

### Decoder-Input Cross-Scale Adapter

Why it is not preferred:

- It behaves like an adapter immediately before decoder input.
- The user wants clearer backbone or neck changes, not a change that is hard to attribute.
- Do not package this as a neck or backbone contribution.

### NeckCtx / ContextRepC3

Run:

```text
uavdetr_r18_neckctx_visdrone640
```

Observed result:

```text
stopped at epoch 32
P: 0.50030
R: 0.36125
AP50: 0.37232
AP50-95: 0.22208
```

Why it failed:

- Lightweight context inside the neck only chased baseline and did not exceed A v1.
- More context alone did not solve small-object separability.
- It did not create a stable recall or AP advantage early enough to justify full training.

## Current Research Direction

Keep the useful part of A v1:

```text
high-resolution small-object feature enhancement around P3
```

Avoid the failed pattern:

```text
broad smoothing, broad low-level detail injection, and query quota changes
```

Next ideas should be selective. A promising module should decide where to enhance small-object detail and where to ignore background texture. It should be a clear backbone or neck change, not a decoder-input patch.

## Early Stopping Lesson

Use same-epoch comparison against baseline and A v1. A run that has no clear advantage by epoch 30-40 can be stopped early if it is a lightweight variant and all main metrics are below baseline/A v1.

For more serious candidates, continue to epoch 100-180 only if at least one target metric has a stable advantage. If a run is below baseline/A v1 by epoch 200-250 and best AP has not improved for about 10 epochs, stop manually.
