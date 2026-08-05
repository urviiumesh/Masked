import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os

# Initialize model
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640))

def load_database():
    db = {}
    for file in os.listdir("faces_db"):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            db[name] = np.load(os.path.join("faces_db", file))
    return db

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face(image_path, db):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image not found: {image_path}")
    
    faces = app.get(img)
    if len(faces) == 0:
        print("No face detected.")
        return None
    
    embedding = faces[0].embedding
    best_match = None
    highest_score = -1
    
    for name, db_emb in db.items():
        # db_emb can be shape (512,) or (N, 512)
        if db_emb.ndim == 1:
            # Single embedding (Legacy)
            score = cosine_similarity(embedding, db_emb)
        else:
            # Multiple embeddings (Multi-View)
            # Calculate similarity with ALL views and take the MAX
            scores = [cosine_similarity(embedding, view) for view in db_emb]
            score = max(scores) if scores else 0.0

        if score > highest_score:
            highest_score = score
            best_match = name

    if highest_score > 0.45:  # Adjust threshold as needed
        print(f"Matched: {best_match} (score={highest_score:.3f})")
    else:
        print("No known face matched.")

if __name__ == "__main__":
    db = load_database()
    image_path = input("Enter image path to recognize: ").strip()
    recognize_face(image_path, db)
