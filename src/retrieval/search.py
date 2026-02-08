import os
import numpy as np
import cv2
from src.models.embedder import FaceEmbedder

GALLERY_PATH = "data/gallery/gallery_embeddings.npy"
GALLERY_IMG_DIR = "data/gallery/photos"
QUERY_IMG = "data/query/001.jpg"
OUTPUT_DIR = "data/output"
TOP_K = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def main():
    if not os.path.exists(GALLERY_PATH):
        print("❌ Gallery not found. Run build_gallery.py first.")
        return

    embedder = FaceEmbedder()

    # Load gallery
    gallery = np.load(GALLERY_PATH, allow_pickle=True).item()
    print(f"📂 Loaded gallery with {len(gallery)} faces")

    if not os.path.exists(QUERY_IMG):
        print("❌ Query image not found:", QUERY_IMG)
        return

    query_emb = embedder.get_embedding(QUERY_IMG)
    if query_emb is None:
        print("❌ No face detected in query image")
        return

    # Compare
    scores = []
    for img_name, emb in gallery.items():
        score = cosine_similarity(query_emb, emb)
        scores.append((img_name, score))

    scores.sort(key=lambda x: x[1], reverse=True)

    print("\n🔍 Top Matches:")
    for i, (name, score) in enumerate(scores[:TOP_K], 1):
        print(f"{i}. {name} | similarity = {score:.4f}")

    # Save best match
    best_match = scores[0][0]
    best_img_path = os.path.join(GALLERY_IMG_DIR, best_match)
    img = cv2.imread(best_img_path)

    if img is not None:
        out_path = os.path.join(OUTPUT_DIR, "best_match.jpg")
        cv2.imwrite(out_path, img)
        print(f"✅ Best match saved to: {out_path}")

if __name__ == "__main__":
    main()
