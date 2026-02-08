import sys
import os
sys.path.append(os.path.abspath("src"))
import cv2
import numpy as np
from models.embedder import FaceEmbedder


GALLERY_DIR = "data/gallery/photos"
OUT_FILE = "data/gallery/gallery_embeddings.npy"

def main():
    embedder = FaceEmbedder()
    gallery = {}

    for fname in os.listdir(GALLERY_DIR):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        path = os.path.join(GALLERY_DIR, fname)
        img = cv2.imread(path)

        if img is None:
            print(f"⚠️ Cannot read image: {path}")
            continue

        emb = embedder.get_embedding(path)

        if emb is None:
            print(f"⚠️ No face detected in: {fname}")
            continue

        gallery[fname] = emb

    np.save(OUT_FILE, gallery)
    print(f"✅ Gallery built with {len(gallery)} faces")

if __name__ == "__main__":
    main()
