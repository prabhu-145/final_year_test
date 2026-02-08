import streamlit as st
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from src.models.embedder import FaceEmbedder

# ---------------- CONFIG ----------------
GALLERY_PATH = "data/gallery/gallery_embeddings.npy"
GALLERY_IMG_DIR = "data/gallery/photos"
TOP_K = 5
THRESHOLD = 0.30

st.set_page_config(
    page_title="Third‑Eye | Forensic Sketch Recognition",
    layout="wide",
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main {
    background-color: #0f172a;
    color: white;
}
h1, h2, h3, h4 {
    color: #e5e7eb;
}
.card {
    background: rgba(255,255,255,0.06);
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 20px;
}
.best {
    border: 2px solid #22c55e;
    box-shadow: 0 0 20px rgba(34,197,94,0.4);
}
.sidebar .sidebar-content {
    background-color: #020617;
}
.stButton>button {
    background: linear-gradient(90deg,#6366f1,#22c55e);
    color: white;
    border-radius: 12px;
    height: 3em;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="card">
<h1 style="text-align:center;">🕵️ Third‑Eye</h1>
<h3 style="text-align:center;color:#9ca3af;">
AI‑Based Forensic Sketch Recognition System
</h3>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")
top_k = st.sidebar.slider("Top‑K Matches", 1, 10, TOP_K)
threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, THRESHOLD)

st.sidebar.markdown("""
---
### 🧠 Tech Stack
- Deep Learning
- InsightFace
- Cosine Similarity
- OpenCV
""")

# ---------------- MAIN ----------------
col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("📤 Upload Sketch / Face")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        with open("temp.jpg", "wb") as f:
            f.write(uploaded_file.read())
        st.image("temp.jpg", caption="Query Image", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PROCESS ----------------
if uploaded_file:
    embedder = FaceEmbedder()
    gallery = np.load(GALLERY_PATH, allow_pickle=True).item()

    query_emb = embedder.get_embedding("temp.jpg")

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("🔍 Matching Results")

        if query_emb is None:
            st.error("❌ No face detected in uploaded image")
        else:
            scores = []
            for name, emb in gallery.items():
                sim = np.dot(query_emb, emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(emb)
                )
                scores.append((name, sim))

            scores.sort(key=lambda x: x[1], reverse=True)
            best_score = scores[0][1]

            if best_score < threshold:
                st.warning("⚠️ No confident match found")
            else:
                # Best match
                best_img = cv2.imread(
                    os.path.join(GALLERY_IMG_DIR, scores[0][0])
                )
                best_img = cv2.cvtColor(best_img, cv2.COLOR_BGR2RGB)

                st.markdown("<div class='card best'>", unsafe_allow_html=True)
                st.markdown("### 🏆 Best Match")
                st.image(
                    best_img,
                    caption=f"{scores[0][0]} | Similarity: {best_score:.3f}",
                    width=280
                )
                st.markdown("</div>", unsafe_allow_html=True)

                # Top‑K list
                st.markdown("### 📊 Top‑K Matches")
                for i, (name, score) in enumerate(scores[:top_k], 1):
                    st.write(f"**{i}. {name}** — `{score:.3f}`")

                # Chart
                names = [x[0] for x in scores[:top_k]]
                values = [x[1] for x in scores[:top_k]]

                fig, ax = plt.subplots()
                ax.barh(names[::-1], values[::-1], color="#6366f1")
                ax.set_xlabel("Cosine Similarity")
                ax.set_xlim(0, 1)

                st.pyplot(fig)

        st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<p style="text-align:center;color:#9ca3af;">
Third‑Eye | Final Year Project | Forensic AI
</p>
""", unsafe_allow_html=True)
