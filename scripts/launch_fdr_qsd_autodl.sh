#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/UAV-DETR-Experiments
RUN=uavdetr_r18_fdr_qsd_visdrone640
MODEL=ultralytics/cfg/models/uavdetr-r18-fdr-qsd.yaml
DATA=/root/autodl-tmp/datasets/VisDrone/visdrone.yaml
LOG=logs/${RUN}.active.log

cd "$ROOT"
mkdir -p logs runs/train

if tmux has-session -t "$RUN" 2>/dev/null; then
  echo "tmux session already exists: $RUN" >&2
  exit 1
fi

if [ -e "runs/train/$RUN" ]; then
  echo "run directory already exists: runs/train/$RUN" >&2
  exit 1
fi

tmux new-session -d -s "$RUN" \
  "cd '$ROOT' && /root/miniconda3/bin/python scripts/train_rtdetr.py \
    --model '$MODEL' \
    --data '$DATA' \
    --imgsz 640 --epochs 400 --batch 4 --workers 8 --device 0 \
    --project '$ROOT/runs/train' --name '$RUN' \
    --patience 20 --optimizer AdamW --lr0 0.0001 --momentum 0.9 \
    --mosaic 1.0 --mixup 0.2 \
    2>&1 | tee '$LOG'"

echo "started tmux=$RUN log=$LOG"
