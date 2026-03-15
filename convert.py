import json
import os
from pathlib import Path

def convert(images_dir):
    for json_file in Path(images_dir).glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        img_w = data["imageWidth"]
        img_h = data["imageHeight"]

        txt_lines = []
        for shape in data["shapes"]:
            if shape["label"] == "sheet":
                pts = shape["points"]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                x_center = ((x_min + x_max) / 2) / img_w
                y_center = ((y_min + y_max) / 2) / img_h
                width    = (x_max - x_min) / img_w
                height   = (y_max - y_min) / img_h

                txt_lines.append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        txt_path = json_file.with_suffix(".txt")
        with open(txt_path, "w") as f:
            f.write("\n".join(txt_lines))
        print(f"Converted: {json_file.name}")

# Запускай для train и val
convert("dataset/images/train")
convert("dataset/images/val")