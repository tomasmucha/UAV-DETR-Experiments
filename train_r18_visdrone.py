import warnings
warnings.filterwarnings("ignore")

import torch
from ultralytics import RTDETR

if __name__ == "__main__":
    torch.cuda.empty_cache()

    model = RTDETR("/root/autodl-tmp/UAV-DETR/ultralytics/cfg/models/uavdetr-r18.yaml")

    model.train(
        data="/root/autodl-tmp/datasets/VisDrone/visdrone.yaml",
        cache=False,
        imgsz=640,
        epochs=400,
        batch=4,
        workers=8,
        device="0",
        project="/root/autodl-tmp/UAV-DETR/runs/train",
        name="uavdetr_r18_visdrone_640",
        patience=20,
        optimizer="AdamW",
        lr0=0.0001,
        momentum=0.9,
        mosaic=1.0,
        mixup=0.2,
    )
