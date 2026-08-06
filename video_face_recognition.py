import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import sys

# Initialize Face Analysis (same as live_recognition.py)
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640))

# Database path for registered faces
REAL_FACES_DB = "faces_db"

# Thresholds tuned for high accuracy (0.35 tuned for mask & sunglasses detection)
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

def process_video(input_path="input_video.mp4", output_path="output_recognized.mp4", display=False):
    if not os.path.isfile(input_path):
        print(f"[ERROR] Input video file not found: {input_path}")
        return

    db = load_database()
    print(f"[INFO] Loaded {len(db)} known face(s) from database: {list(db.keys())}")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {input_path}")
        return

    # Video parameters
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    print(f"[INFO] Processing {total_frames} frames from '{input_path}'...")

    dirty_faces = set()
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % 30 == 0 or frame_idx == total_frames:
                print(f"  Processing frame {frame_idx}/{total_frames}...")

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

                    # Safe Online Learning: Only add view if match confidence is very high (>= 0.55)
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

            out.write(frame)

            if display:
                cv2.imshow("Video Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

    print(f"\n[SUCCESS] Processed video saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        in_path = sys.argv[1]
    else:
        in_path = "input_video.mp4"

    out_path = sys.argv[2] if len(sys.argv) > 2 else "output_recognized.mp4"
    process_video(in_path, out_path)
