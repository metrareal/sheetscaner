# import cv2
# import numpy as np
# from ultralytics import YOLO

# # Загрузка модели и изображений
# # model = YOLO("runs/detect/sheet_detector3/weights/best.pt")
# model = YOLO("runs/detect/sheet_detector_gen/weights/best.pt")
# conveyor = cv2.imread("conveyor.jpg")
# sheet = cv2.imread("sheet.jpg")

# # Размеры окна
# WIN_W, WIN_H = 1280, 480
# conveyor = cv2.resize(conveyor, (WIN_W, WIN_H))

# # Размер листа — делаем его меньше окна
# SHEET_H = int(WIN_H * 0.5)
# SHEET_W = int(sheet.shape[1] * SHEET_H / sheet.shape[0])
# sheet = cv2.resize(sheet, (SHEET_W, SHEET_H))

# # Позиция листа
# sheet_x = -SHEET_W  # начинает за левым краем
# sheet_y = (WIN_H - SHEET_H) // 2  # по центру по высоте
# speed = 3  # пикселей за кадр

# # Список замеров ширины
# measurements = []
# step = 20  # каждые 20 пикселей замеряем

# def measure_width(frame, x, y, w, h):
#     """Измеряем ширину листа через YOLO bounding box"""
#     return h  # пока берём высоту bounding box как ширину

# def draw_contour(measurements):
#     """Рисуем схематический контур листа"""
#     if len(measurements) < 2:
#         return np.ones((300, 800, 3), dtype=np.uint8) * 255

#     canvas = np.ones((300, 800, 3), dtype=np.uint8) * 255
#     max_w = max(measurements) if measurements else 1

#     points_top = []
#     points_bot = []

#     for i, w in enumerate(measurements):
#         x = int(i * 800 / len(measurements))
#         center_y = 150
#         half = int(w * 100 / max_w)
#         points_top.append((x, center_y - half))
#         points_bot.append((x, center_y + half))

#     # Рисуем контур
#     for i in range(1, len(points_top)):
#         cv2.line(canvas, points_top[i-1], points_top[i], (0, 0, 255), 2)
#         cv2.line(canvas, points_bot[i-1], points_bot[i], (0, 0, 255), 2)

#     # Подписи ширины каждые 10 замеров
#     for i in range(0, len(measurements), 10):
#         x = int(i * 800 / len(measurements))
#         cv2.putText(canvas, f"{measurements[i]}px",
#                     (x, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

#     cv2.putText(canvas, "Contour of sheet", (10, 20),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
#     return canvas

# print("Запуск прототипа... Нажми Q для выхода")

# while True:
#     # Фон конвейера
#     frame = conveyor.copy()

#     # Координаты листа в кадре
#     x1 = max(sheet_x, 0)
#     x2 = min(sheet_x + SHEET_W, WIN_W)
#     sx1 = x1 - sheet_x
#     sx2 = sx1 + (x2 - x1)

#     sheet_visible = x2 > x1 and sx2 > sx1

#     if sheet_visible:
#         # Накладываем лист на фон
#         frame[sheet_y:sheet_y+SHEET_H, x1:x2] = sheet[0:SHEET_H, sx1:sx2]

#         # YOLO детекция
#         results = model(frame, verbose=False)

#         sheet_detected = False
#         for r in results:
#             for box in r.boxes:
#                 bx1, by1, bx2, by2 = map(int, box.xyxy[0])
#                 conf = float(box.conf[0])
#                 # print(f"conf={conf:.3f}")

#                 if conf > 0.25:
#                     sheet_detected = True
#                     # Рисуем bounding box
#                     cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
#                     cv2.putText(frame, f"sheet {conf:.2f}",
#                                 (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX,
#                                 0.6, (0, 255, 0), 2)

#                     # Замеряем ширину каждые step пикселей
#                     if sheet_x > 0 and sheet_x % step == 0:
#                         width = by2 - by1
#                         measurements.append(width)

#                         # Рисуем вертикальную линию замера
#                         mid_x = (bx1 + bx2) // 2
#                         cv2.line(frame, (mid_x, by1), (mid_x, by2), (0, 0, 255), 1)
#                         cv2.putText(frame, f"{width}px",
#                                     (mid_x+5, (by1+by2)//2),
#                                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

#         if not sheet_detected:
#             cv2.putText(frame, "Searching...", (20, 40),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
#     else:
#         cv2.putText(frame, "Waiting for sheet...", (20, 40),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

#     # Двигаем лист
#     sheet_x += speed

#     # Лист вышел за правый край — стоп
#     if sheet_x > WIN_W:
#         print(f"Лист прошёл. Замеров: {len(measurements)}")
#         break

#     # Показываем окно конвейера
#     cv2.imshow("Conveyor", frame)

#     # Показываем контур
#     contour_img = draw_contour(measurements)
#     cv2.imshow("Sheet Contour", contour_img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Финальный контур
# print("Замеры ширины:", measurements)
# contour_img = draw_contour(measurements)
# cv2.imshow("Sheet Contour - Final", contour_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

model = YOLO("runs/detect/sheet_detector_gen/weights/best.pt")
conveyor_orig = cv2.imread("conveyor.jpg")
sheet_orig = cv2.imread("sheet.jpg")

WIN_W, WIN_H = 1280, 480
conveyor_bg = cv2.resize(conveyor_orig, (WIN_W, WIN_H))

SHEET_H = int(WIN_H * 0.5)
SHEET_W = int(sheet_orig.shape[1] * SHEET_H / sheet_orig.shape[0])
sheet_img = cv2.resize(sheet_orig, (SHEET_W, SHEET_H))

sheet_x = -SHEET_W
sheet_y = (WIN_H - SHEET_H) // 2
speed = 3

# Замеры: список (x_позиция, ширина_в_пикселях)
measurements = []

# Сглаживание — скользящее среднее
smooth_window = deque(maxlen=5)

def get_real_edges(frame, bx1, by1, bx2, by2, conveyor_bg):
    """Находим реальные верхний и нижний край листа внутри bounding box"""
    roi_frame = frame[by1:by2, bx1:bx2].astype(np.float32)
    roi_bg = conveyor_bg[by1:by2, bx1:bx2].astype(np.float32)

    # Разница между кадром и фоном
    diff = cv2.absdiff(roi_frame, roi_bg)
    diff_gray = cv2.cvtColor(diff.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Бинаризация
    _, mask = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)

    # Морфология — убираем шум
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Находим верхний и нижний край по каждой колонке
    top_edges = []
    bot_edges = []
    h, w = mask.shape

    for col in range(w):
        col_pixels = np.where(mask[:, col] > 0)[0]
        if len(col_pixels) > 0:
            top_edges.append(by1 + col_pixels[0])
            bot_edges.append(by1 + col_pixels[-1])
        else:
            top_edges.append(None)
            bot_edges.append(None)

    return top_edges, bot_edges

def draw_contour(measurements, sheet_w=SHEET_W, sheet_h=SHEET_H):
    """Рисуем схематический контур листа"""
    if len(measurements) < 2:
        canvas = np.ones((300, 800, 3), dtype=np.uint8) * 255
        cv2.putText(canvas, "Waiting for data...", (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 150), 2)
        return canvas

    canvas = np.ones((300, 800, 3), dtype=np.uint8) * 255

    widths = [m[1] for m in measurements]
    max_w = max(widths) if widths else 1
    min_w = min(widths)
    avg_w = int(np.mean(widths))

    points_top = []
    points_bot = []
    n = len(measurements)

    # Реальная пропорция: ширина контура = высота * (B/A)
    avg_h = avg_w  # высота в пикселях
    real_ratio = SHEET_W / SHEET_H  # соотношение сторон листа
    contour_w = min(int(200 * real_ratio), 780)  # ширина контура на canvas
    step_px = contour_w / n if n > 1 else 1

    for i, (_, w) in enumerate(measurements):
        # x = int(i * 780 / n) + 10
        x = int(i * step_px) + 10
        center_y = 150
        half = int(w * 100 / max_w)
        points_top.append((x, center_y - half))
        points_bot.append((x, center_y + half))

    # Рисуем контур
    for i in range(1, len(points_top)):
        cv2.line(canvas, points_top[i-1], points_top[i], (0, 0, 200), 2)
        cv2.line(canvas, points_bot[i-1], points_bot[i], (0, 0, 200), 2)

    # Закрываем контур слева и справа
    if points_top and points_bot:
        cv2.line(canvas, points_top[0], points_bot[0], (0, 0, 200), 2)
        cv2.line(canvas, points_top[-1], points_bot[-1], (0, 0, 200), 2)

    # Подписи
    cv2.putText(canvas, f"Max: {max_w}px  Min: {min_w}px  Avg: {avg_w}px",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    cv2.putText(canvas, f"Measurements: {n}",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # Числа ширины каждые 15 замеров
    for i in range(0, len(measurements), 15):
        x = int(i * 780 / n) + 10
        cv2.putText(canvas, f"{measurements[i][1]}",
                    (x, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1)

    return canvas

print("Запуск... Нажми Q для выхода")

while True:
    frame = conveyor_bg.copy()

    x1 = max(sheet_x, 0)
    x2 = min(sheet_x + SHEET_W, WIN_W)
    sx1 = x1 - sheet_x
    sx2 = sx1 + (x2 - x1)

    sheet_visible = x2 > x1 and sx2 > sx1

    if sheet_visible:
        frame[sheet_y:sheet_y+SHEET_H, x1:x2] = sheet_img[0:SHEET_H, sx1:sx2]

        results = model(frame, verbose=False)

        for r in results:
            for box in r.boxes:
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                if conf > 0.25:
                    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    cv2.putText(frame, f"sheet {conf:.2f}",
                                (bx1, by1-10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)

                    # Реальные края листа
                    top_edges, bot_edges = get_real_edges(
                        frame, bx1, by1, bx2, by2, conveyor_bg)

                    # Замеряем ширину каждые step пикселей
                    if sheet_x > 0 and sheet_x % 20 == 0:
                        valid = [(t, b) for t, b in zip(top_edges, bot_edges)
                                 if t is not None and b is not None]
                        if valid:
                            avg_width = int(np.mean([b - t for t, b in valid]))
                            smooth_window.append(avg_width)
                            smoothed = int(np.mean(smooth_window))
                            measurements.append((sheet_x, smoothed))

                            # Вертикальная линия замера в центре
                            mid_x = (bx1 + bx2) // 2
                            cv2.line(frame, (mid_x, by1), (mid_x, by2),
                                     (255, 0, 0), 2)
                            cv2.putText(frame, f"{smoothed}px",
                                        (mid_x+5, (by1+by2)//2),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.4, (255, 0, 0), 1)

    else:
        cv2.putText(frame, "Waiting for sheet...", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    sheet_x += speed

    if sheet_x > WIN_W:
        print(f"Лист прошёл. Замеров: {len(measurements)}")
        break

    cv2.imshow("Conveyor", frame)
    contour_img = draw_contour(measurements)
    cv2.imshow("Sheet Contour", contour_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Финальный результат
print("Замеры ширины:", [m[1] for m in measurements])
contour_img = draw_contour(measurements)
cv2.imshow("Sheet Contour - Final", contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()