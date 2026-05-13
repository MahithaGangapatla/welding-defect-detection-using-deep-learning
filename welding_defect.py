from ultralytics import YOLO
import cv2
import os
import numpy as np

# =========================
# STEP 1: TRAIN (CLASSIFICATION)
# =========================

print("🔵 Training started...")

model = YOLO("yolov8n-cls.pt")

model.train(
    data="dataset",   # your 8 class folders
    epochs=8,
    imgsz=160,
    batch=16,
    name="auto_cls"
)

print("✅ Training completed")

# =========================
# STEP 2: LOAD MODEL
# =========================

model = YOLO("runs/classify/auto_cls/weights/best.pt")

# =========================
# STEP 3: PREDICT + SORT
# =========================

input_root = "dataset"
output_root = "output_final"

os.makedirs(output_root, exist_ok=True)

print("🔵 Detection started...")

for cls in os.listdir(input_root):
    cls_path = os.path.join(input_root, cls)

    for img_name in os.listdir(cls_path):
        img_path = os.path.join(cls_path, img_name)

        img = cv2.imread(img_path)
        if img is None:
            continue

        result = model(img)[0]

        pred = result.names[result.probs.top1]
        conf = float(result.probs.top1conf)

        # 🔥 bounding box (full image demo)
        h, w = img.shape[:2]
        cv2.rectangle(img, (0,0), (w,h), (0,255,0), 2)

        cv2.putText(img, f"{pred} {conf:.2f}",
                    (20,40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        # 🔥 correct folder
        save_folder = os.path.join(output_root, pred)
        os.makedirs(save_folder, exist_ok=True)

        cv2.imwrite(os.path.join(save_folder, img_name), img)

print("🔥 DONE — CHECK output_final")