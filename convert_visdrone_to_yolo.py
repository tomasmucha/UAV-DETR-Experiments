from pathlib import Path
from PIL import Image

ROOT = Path("/root/autodl-tmp/datasets/VisDrone")

splits = [
    ROOT / "VisDrone2019-DET-train",
    ROOT / "VisDrone2019-DET-val",
]

for split_dir in splits:
    img_dir = split_dir / "images"
    ann_dir = split_dir / "annotations"
    label_dir = split_dir / "labels"
    label_dir.mkdir(parents=True, exist_ok=True)

    count_img = 0
    count_box = 0

    for img_path in sorted(img_dir.glob("*.jpg")):
        count_img += 1
        w, h = Image.open(img_path).size
        ann_path = ann_dir / f"{img_path.stem}.txt"
        out_path = label_dir / f"{img_path.stem}.txt"

        lines_out = []
        if ann_path.exists():
            for line in ann_path.read_text().strip().splitlines():
                parts = line.split(",")
                if len(parts) < 6:
                    continue

                x, y, bw, bh = map(float, parts[:4])
                score = int(parts[4])
                cls = int(parts[5])

                if score == 0:
                    continue
                if cls < 1 or cls > 10:
                    continue
                if bw <= 0 or bh <= 0:
                    continue

                cls_id = cls - 1
                xc = (x + bw / 2) / w
                yc = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h

                xc = min(max(xc, 0), 1)
                yc = min(max(yc, 0), 1)
                nw = min(max(nw, 0), 1)
                nh = min(max(nh, 0), 1)

                lines_out.append(f"{cls_id} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
                count_box += 1

        out_path.write_text("\n".join(lines_out))

    print(f"{split_dir.name}: images={count_img}, boxes={count_box}")
