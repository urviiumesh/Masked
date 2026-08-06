import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os

# Initialize InsightFace model
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640))

# Database path for known faces
REAL_FACES_DB = "faces_db"

# Thresholds tuned for high accuracy (0.35 tuned for masks & sunglasses detection)
RECOGNITION_THRESHOLD = 0.35  # Minimum similarity score to match a known face
HIGH_CONF_THRESHOLD = 0.55   # Minimum similarity score to safely learn new view in memory

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

# Load registered known faces
db = load_database()
print(f"Loaded {len(db)} known face(s) from database: {list(db.keys())}")

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Could not open webcam. Check if another app is using it.")

print("Press 'q' to quit the live recognition window.")

# Create named window for display
cv2.namedWindow("Live Face Recognition", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Live Face Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

dirty_faces = set()

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Detect faces from the frame
        faces = app.get(frame)
        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int)
            emb = face.embedding

            # Find best match in known database
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

            # Only draw bounding box and label if recognized as a KNOWN face
            if best_match and best_score >= RECOGNITION_THRESHOLD:
                color = (0, 255, 0)  # Green box for known face
                label = f"{best_match} ({best_score:.2f})"

                # Safe Online Learning: Only add view if match confidence is very high (> 0.55)
                # This prevents non-matches or noisy angles from polluting known embeddings
                if best_score >= HIGH_CONF_THRESHOLD:
                    current_db_emb = db[best_match]
                    if current_db_emb.ndim == 1:
                        current_db_emb = np.expand_dims(current_db_emb, axis=0)

                    updated_emb = np.vstack([current_db_emb, emb])
                    if len(updated_emb) > 50:
                        updated_emb = updated_emb[-50:]

                    db[best_match] = updated_emb
                    dirty_faces.add(best_match)

                # Draw bounding box and label ONLY for known faces
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            # If unknown (best_score < RECOGNITION_THRESHOLD), no bounding box is drawn at all!

        cv2.imshow("Live Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Cleanup camera and windows (Disk .npy files remain clean & protected)
    cap.release()
    cv2.destroyAllWindows()
