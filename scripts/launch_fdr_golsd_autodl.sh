#!/usr/bin/env bash
set -euo pipefail

project=/root/autodl-tmp/UAV-DETR-Experiments
run=uavdetr_r18_fdr_golsd_visdrone640
session=${run}

cd "${project}"
mkdir -p logs runs/train

if tmux has-session -t "${session}" 2>/dev/null; then
    echo "tmux session already exists: ${session}" >&2
    exit 1
fi
if [[ -e "runs/train/${run}" ]]; then
    echo "run directory already exists: runs/train/${run}" >&2
    exit 1
fi

tmux new-session -d -s "${session}" bash -lc "
set -o pipefail
cd '${project}'
/root/miniconda3/bin/python scripts/train_rtdetr.py \
  --model ultralytics/cfg/models/uavdetr-r18-fdr-golsd.yaml \
  --data /root/autodl-tmp/datasets/VisDrone/visdrone.yaml \
  --imgsz 640 --epochs 400 --batch 4 --workers 8 --device 0 \
  --project runs/train --name '${run}' \
  --patience 20 --optimizer AdamW --lr0 0.0001 --momentum 0.9 \
  --mosaic 1.0 --mixup 0.2 \
  2>&1 | tee 'logs/${run}.active.log'
"

tmux has-session -t "${session}"
echo "started ${session}"
