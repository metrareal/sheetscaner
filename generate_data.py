import cv2
import numpy as np
import os
import random
from pathlib import Path

# Загрузка изображений
conveyor = cv2.imread("conveyor.jpg")
sheet = cv2.imread("sheet.jpg")

# Размеры окна
WIN_W, WIN_H = 1280, 480
conveyor = cv2.resize(conveyor, (WIN_W, WIN_H))

# Размер листа
SHEET_H = int(WIN_H * 0.5)
SHEET_W = int(sheet.shape[1] * SHEET_H / sheet.shape[0])
sheet = cv2.resize(sheet, (SHEET_W, SHEET_H))

# Папки для сохранения
for folder in ["dataset/images/train", "dataset/images/val",
               "dataset/labels/train", "dataset/labels/val"]:
    Path(folder).mkdir(parents=True, exist_ok=True)

count = 0

for i in range(60):  # генерируем 60 изображений
    frame = conveyor.copy()

    # Случайная позиция листа по X
    sheet_x = random.randint(0, WIN_W - SHEET_W)
    sheet_y = (WIN_H - SHEET_H) // 2

    # Накладываем лист
    x1 = sheet_x
    x2 = sheet_x + SHEET_W
    frame[sheet_y:sheet_y+SHEET_H, x1:x2] = sheet

    # YOLO разметка (автоматически)
    x_center = (x1 + x2) / 2 / WIN_W
    y_center = (sheet_y + sheet_y + SHEET_H) / 2 / WIN_H
    width = SHEET_W / WIN_W
    height = SHEET_H / WIN_H

    label = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

    # 80% train, 20% val
    if i < 48:
        folder = "train"
    else:
        folder = "val"

    img_path = f"dataset/images/{folder}/gen_{i:03d}.jpg"
    lbl_path = f"dataset/labels/{folder}/gen_{i:03d}.txt"

    cv2.imwrite(img_path, frame)
    with open(lbl_path, "w") as f:
        f.write(label)

    count += 1

print(f"Сгенерировано {count} изображений")