import os
import cv2
import numpy as np

INPUT_SKETCH = "data/cleaned/sketches/sketches"
INPUT_PHOTO = "data/cleaned/photos/photo"

OUTPUT_SKETCH = "data/aligned/sketches"
OUTPUT_PHOTO = "data/aligned/photos"

MODEL_PROTO = "models/opencv/deploy.prototxt"
MODEL_WEIGHTS = "models/opencv/res10_300x300_ssd_iter_140000.caffemodel"

os.makedirs(OUTPUT_SKETCH, exist_ok=True)
os.makedirs(OUTPUT_PHOTO, exist_ok=True)

net = cv2.dnn.readNetFromCaffe(MODEL_PROTO, MODEL_WEIGHTS)

def resize_only(input_dir, output_dir):
    for img_name in os.listdir(input_dir):
        print("Processing sketch:", img_name)
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (256, 256))
        cv2.imwrite(os.path.join(output_dir, img_name), img)
        print(f"Resized sketch: {img_name}")

def detect_and_align(input_dir, output_dir):
    for img_name in os.listdir(input_dir):
        print("Processing photo:", img_name)
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        net.setInput(blob)
        detections = net.forward()

        if detections.shape[2] == 0:
            print(f"No face: {img_name}")
            continue

        confidence = detections[0, 0, 0, 2]
        if confidence < 0.5:
            print(f"Low confidence face: {img_name}")
            continue

        box = detections[0, 0, 0, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        face = img[y1:y2, x1:x2]
        face = cv2.resize(face, (256, 256))

        cv2.imwrite(os.path.join(output_dir, img_name), face)
        print(f"Aligned photo: {img_name}")

# Run pipeline
resize_only(INPUT_SKETCH, OUTPUT_SKETCH)
detect_and_align(INPUT_PHOTO, OUTPUT_PHOTO)

print("✅ Phase 2 completed")
