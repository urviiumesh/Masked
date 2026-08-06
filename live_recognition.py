import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import shutil

# Initialize InsightFace model
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640))

# Database paths
REAL_FACES_DB = "faces_db"               # Main local database store
ACTIVE_TARGETS_DB = "selected_targets_db" # Active target embeddings folder

# Thresholds tuned for high accuracy (0.35 tuned for masks & sunglasses detection)
RECOGNITION_THRESHOLD = 0.35  # Minimum similarity score to match a known face
HIGH_CONF_THRESHOLD = 0.55   # Minimum similarity score to safely learn new view in memory

def select_target_database(target_input=None):
    """Prompt user for target names, copy embeddings to selected_targets_db, and load them."""
    if target_input is None:
        target_input = input("\nEnter person name(s) to recognize (comma-separated, e.g., Vidit, Urvi, Pranav, or press Enter for ALL): ").strip()

    os.makedirs(ACTIVE_TARGETS_DB, exist_ok=True)

    # Clear previous files in ACTIVE_TARGETS_DB
    for f in os.listdir(ACTIVE_TARGETS_DB):
        fp = os.path.join(ACTIVE_TARGETS_DB, f)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except Exception:
                pass

    db = {}
    if not target_input or target_input.lower() == 'all':
        print(f"\n[INFO] No specific target selected. Loading ALL profiles from '{REAL_FACES_DB}'...")
        if os.path.exists(REAL_FACES_DB):
            for file in os.listdir(REAL_FACES_DB):
                if file.endswith(".npy"):
                    name = file.replace(".npy", "")
                    db[name] = np.load(os.path.join(REAL_FACES_DB, file))
        print(f"[INFO] Loaded {len(db)} total profile(s) into search space.")
        return db

    # Parse requested target names
    raw_targets = [t.strip() for t in target_input.split(",") if t.strip()]
    found_targets = []

    for t_name in raw_targets:
        src_npy = os.path.join(REAL_FACES_DB, f"{t_name}.npy")
        if os.path.isfile(src_npy):
            dest_npy = os.path.join(ACTIVE_TARGETS_DB, f"{t_name}.npy")
            shutil.copy2(src_npy, dest_npy)
            db[t_name] = np.load(dest_npy)
            found_targets.append(t_name)
        else:
            print(f"[WARNING] Target '{t_name}' not found in '{REAL_FACES_DB}'!")

    if not db:
        print(f"[ERROR] None of the specified targets {raw_targets} were found in '{REAL_FACES_DB}'!")
    else:
        print(f"\n[SUCCESS] Copied and loaded {len(found_targets)} target profile(s) into '{ACTIVE_TARGETS_DB}': {found_targets}")
        print(f"[INFO] System will ONLY detect and recognize these selected target(s)!")

    return db

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Prompt user for target names and load selected database
db = select_target_database()

# Start webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit("Could not open webcam. Check if another app is using it.")

print("\nPress 'q' to quit the live recognition window.")

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

            # Find best match in target database
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

            # Only draw bounding box and label if recognized as a SELECTED target face
            if best_match and best_score >= RECOGNITION_THRESHOLD:
                color = (0, 255, 0)  # Green box for known target face
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

                # Draw bounding box and label ONLY for selected target faces
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Live Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Save updated multi-view embeddings to both active target folder and main local database
    if dirty_faces:
        for name in dirty_faces:
            if name in db:
                try:
                    # Save to main database
                    main_npy = os.path.join(REAL_FACES_DB, f"{name}.npy")
                    np.save(main_npy, db[name])

                    # Save to active targets folder if present
                    target_npy = os.path.join(ACTIVE_TARGETS_DB, f"{name}.npy")
                    if os.path.exists(ACTIVE_TARGETS_DB):
                        np.save(target_npy, db[name])

                    print(f"Saved updated multi-view embeddings for '{name}' to main database.")
                except Exception as e:
                    print(f"Failed to save embeddings for {name}: {e}")

    cap.release()
    cv2.destroyAllWindows()
