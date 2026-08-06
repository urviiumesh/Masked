import cv2
import numpy as np
from insightface.app import FaceAnalysis
import os
import sys
import shutil

# Initialize Face Analysis (same as live_recognition.py)
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640))

# Database paths
REAL_FACES_DB = "faces_db"               # Main local database store
ACTIVE_TARGETS_DB = "selected_targets_db" # Active target embeddings folder

# Thresholds tuned for high accuracy (0.35 tuned for mask & sunglasses detection)
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

def process_video(input_path="input_video.mp4", output_path="output_recognized.mp4", display=False, target_names=None):
    if not os.path.isfile(input_path):
        print(f"[ERROR] Input video file not found: {input_path}")
        return

    # Select target database
    db = select_target_database(target_names)
    if not db:
        print("[ERROR] Cannot proceed without a valid target database.")
        return

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

                    # Draw bounding box and label ONLY for target faces
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            out.write(frame)

            if display:
                cv2.imshow("Video Face Recognition", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    finally:
        # Save updated multi-view embeddings for registered target profiles
        if dirty_faces:
            print(f"[INFO] Saving updated embeddings for {len(dirty_faces)} target profile(s)...")
            for name in dirty_faces:
                if name in db:
                    try:
                        main_npy = os.path.join(REAL_FACES_DB, f"{name}.npy")
                        np.save(main_npy, db[name])

                        target_npy = os.path.join(ACTIVE_TARGETS_DB, f"{name}.npy")
                        if os.path.exists(ACTIVE_TARGETS_DB):
                            np.save(target_npy, db[name])

                        print(f"  Saved updated embeddings for '{name}'.")
                    except Exception as e:
                        print(f"  Failed to save embeddings for {name}: {e}")

        cap.release()
        out.release()
        if display:
            cv2.destroyAllWindows()

    print(f"\n[SUCCESS] Processed video saved to: {output_path}")

if __name__ == "__main__":
    in_path = sys.argv[1] if len(sys.argv) > 1 else "input_video.mp4"
    out_path = sys.argv[2] if len(sys.argv) > 2 else "output_recognized.mp4"
    targets = sys.argv[3] if len(sys.argv) > 3 else None
    process_video(in_path, out_path, target_names=targets)
