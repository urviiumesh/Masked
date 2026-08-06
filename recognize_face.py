import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import sys

# Initialize InsightFace model
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640))

# Database path for known faces
REAL_FACES_DB = "faces_db"

# Recognition threshold (tuned for mask & sunglasses detection)
RECOGNITION_THRESHOLD = 0.35

def load_database():
    db = {}
    if os.path.exists(REAL_FACES_DB):
        for file in os.listdir(REAL_FACES_DB):
            if file.endswith(".npy"):
                name = file.replace(".npy", "")
                db[name] = np.load(os.path.join(REAL_FACES_DB, file))
    return db

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face_in_image(image_path, db):
    if not os.path.isfile(image_path):
        print(f"Error: Image file not found -> {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image -> {image_path}")
        return

    faces = app.get(img)
    if len(faces) == 0:
        print("No face detected.")
        return

    any_matched = False
    for i, face in enumerate(faces):
        emb = face.embedding
        best_match = None
        best_score = 0.0

        for name, db_emb in db.items():
            if db_emb.ndim == 1:
                score = cosine_similarity(emb, db_emb)
            else:
                scores = [cosine_similarity(emb, view) for view in db_emb]
                score = max(scores) if scores else 0.0

            if score > best_score:
                best_score = score
                best_match = name

        if best_match and best_score >= RECOGNITION_THRESHOLD:
            any_matched = True
            confidence_pct = best_score * 100.0
            print(f"Name: {best_match} | Similarity Score: {best_score:.3f} ({confidence_pct:.1f}%)")
        else:
            print(f"Face {i+1}: Unrecognized / Below Threshold (Best Candidate: {best_match}, Score: {best_score:.3f})")

    if not any_matched:
        print("No known face matched.")

if __name__ == "__main__":
    db = load_database()
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter image path to recognize: ").strip()

    image_path = image_path.strip('"').strip("'")
    recognize_face_in_image(image_path, db)
