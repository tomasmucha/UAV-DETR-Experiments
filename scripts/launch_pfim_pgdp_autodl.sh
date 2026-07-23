#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp/UAV-DETR-Experiments}"
EXPECTED_COMMIT="${EXPECTED_COMMIT:?Set EXPECTED_COMMIT to the pushed experiment commit}"
RUN=uavdetr_r18_pfim_pgdp_visdrone640
MODEL=ultralytics/cfg/models/uavdetr-r18-pfim-pgdp.yaml
DATA=/root/autodl-tmp/datasets/VisDrone/visdrone.yaml
LOG=logs/${RUN}.active.log

cd "$ROOT"

if [ "$(git rev-parse HEAD)" != "$EXPECTED_COMMIT" ]; then
  echo "unexpected worktree commit: $(git rev-parse HEAD)" >&2
  exit 1
fi

if [ -n "$(git status --porcelain)" ]; then
  echo "worktree is not clean: $ROOT" >&2
  exit 1
fi

test -f "$MODEL"
test -f "$DATA"
mkdir -p logs runs/train

if tmux has-session -t "$RUN" 2>/dev/null; then
  echo "tmux session already exists: $RUN" >&2
  exit 1
fi

if [ -e "runs/train/$RUN" ]; then
  echo "run directory already exists: runs/train/$RUN" >&2
  exit 1
fi

TRAIN_CODE="from ultralytics import RTDETR; RTDETR('$MODEL').train(data='$DATA', imgsz=640, epochs=400, batch=4, workers=8, device='0', project='$ROOT/runs/train', name='$RUN', patience=20, cache=False, fraction=1.0, val=True, exist_ok=False, optimizer='AdamW', lr0=0.0001, momentum=0.9, mosaic=1.0, mixup=0.2, seed=0, deterministic=True, amp=False)"

tmux new-session -d -s "$RUN" \
  "cd '$ROOT' && /root/miniconda3/bin/python -c \"$TRAIN_CODE\" 2>&1 | tee '$LOG'"

echo "started tmux=$RUN seed=0 commit=$EXPECTED_COMMIT log=$LOG"
