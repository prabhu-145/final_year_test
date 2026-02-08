import cv2
from insightface.app import FaceAnalysis

class FaceEmbedder:
    def __init__(self):
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_embedding(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return None

        faces = self.app.get(img)
        if len(faces) == 0:
            return None

        return faces[0].embedding
