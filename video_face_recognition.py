import cv2
import insightface
from insightface.app import FaceAnalysis
import numpy as np
import os
import time
from register_face import register_face

# Initialize Face Analysis
app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

REAL_FACES_DB = "faces_db"
TEMP_DB_ROOT = "temp_face_database"
TEMP_EMB_ROOT = "temp_faces_db"

# Ensure temp directories exist
os.makedirs(TEMP_DB_ROOT, exist_ok=True)
os.makedirs(TEMP_EMB_ROOT, exist_ok=True)

# Global database
db = {}

def load_database():
    global db
    db = {}
    
    # Load real faces
    if os.path.exists(REAL_FACES_DB):
        for file in os.listdir(REAL_FACES_DB):
            if file.endswith(".npy"):
                name = file.replace(".npy", "")
                db[name] = np.load(os.path.join(REAL_FACES_DB, file))
    
    # Load temp faces
    if os.path.exists(TEMP_EMB_ROOT):
        for file in os.listdir(TEMP_EMB_ROOT):
            if file.endswith(".npy"):
                name = file.replace(".npy", "")
                db[name] = np.load(os.path.join(TEMP_EMB_ROOT, file))
    
    print(f"Loaded {len(db)} faces from database")

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face(face_embedding):
    best_match = "Unknown"
    best_score = 0.0
    
    for name, db_emb in db.items():
        if db_emb.ndim == 1:
            score = cosine_similarity(face_embedding, db_emb)
        else:
            scores = [cosine_similarity(face_embedding, view) for view in db_emb]
            score = max(scores) if scores else 0.0

        if score > best_score:
            best_score = score
            best_match = name
            
    # Threshold
    # Dynamic Thresholding is handled in the main loop now, but we return the best match here
    # We can just return the match and let the loop decide based on the name
    return best_match, best_score

def get_next_unknown_id():
    # Find all folders starting with "unknown_"
    existing = [d for d in os.listdir(TEMP_DB_ROOT) if os.path.isdir(os.path.join(TEMP_DB_ROOT, d)) and d.startswith("unknown_")]
    if not existing:
        return 1
    
    # Extract numbers
    ids = []
    for d in existing:
        try:
            ids.append(int(d.split("_")[1]))
        except (IndexError, ValueError):
            pass
    
    return max(ids) + 1 if ids else 1

# Initial load
load_database()

# Video paths
input_path = "input_video.mp4"
output_path = "output_recognized.mp4"

cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise Exception(f"Error opening video file {input_path}")

# Setup video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Processing {frame_count} frames...")

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        best_match, score = recognize_face(face.normed_embedding)
        
        # Dynamic Thresholding
        if best_match.startswith("unknown"):
            threshold = 0.35
        else:
            # 0.30 is better to avoid false positives (0.25 was too lenient)
            threshold = 0.30
            
        if score > threshold:
            name = best_match
        else:
            name = "Unknown"
        
        # ONLINE LEARNING
        if name != "Unknown":
            # Update the database with this new view
            if name in db:
                current_db_emb = db[name]
                if current_db_emb.ndim == 1:
                    current_db_emb = np.expand_dims(current_db_emb, axis=0)
                
                # Add new embedding
                updated_emb = np.vstack([current_db_emb, face.normed_embedding])
                
                # Limit to last 50 embeddings
                if len(updated_emb) > 50:
                    updated_emb = updated_emb[-50:]
                
                db[name] = updated_emb
                
                # If it's an unknown person, persist to disk
                if name.startswith("unknown"):
                    try:
                        npy_path = os.path.join(TEMP_EMB_ROOT, f"{name}.npy")
                        np.save(npy_path, updated_emb)
                    except Exception as e:
                        print(f"Failed to update persistent DB for {name}: {e}")
        
        if name == "Unknown":
             # Handle new unknown face
            # print(f"New unknown face detected (score: {score:.2f})")
            
            # Crop the face with padding
            h, w, _ = frame.shape
            pad_w = int((x2 - x1) * 0.25)
            pad_h = int((y2 - y1) * 0.25)
            crop_x1 = max(0, x1 - pad_w)
            crop_y1 = max(0, y1 - pad_h)
            crop_x2 = min(w, x2 + pad_w)
            crop_y2 = min(h, y2 + pad_h)
            
            face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # STRICT VALIDATION: Check if crop actually has a face
            crop_faces = app.get(face_crop)
            if len(crop_faces) == 0:
                print(f"Skipping unknown registration: No face detected in crop (False Positive).")
                continue
                
            # RE-VERIFICATION: Check if crop embedding matches anyone
            crop_emb = crop_faces[0].embedding
            check_match, check_score = recognize_face(crop_emb)
            
            # Check thresholds again for the crop
            check_threshold = 0.35 if check_match.startswith("unknown") else 0.30
            
            if check_score > check_threshold:
                print(f"Crop matched {check_match} ({check_score:.2f})! Updating instead of registering new.")
                
                # It's a match! Update DB (Online Learning)
                if check_match in db:
                    current_db_emb = db[check_match]
                    if current_db_emb.ndim == 1:
                        current_db_emb = np.expand_dims(current_db_emb, axis=0)
                    updated_emb = np.vstack([current_db_emb, crop_emb])
                    if len(updated_emb) > 50:
                        updated_emb = updated_emb[-50:]
                    db[check_match] = updated_emb
                    
                    if check_match.startswith("unknown"):
                        try:
                            np.save(os.path.join(TEMP_EMB_ROOT, f"{check_match}.npy"), updated_emb)
                        except: pass
                
                # Draw label for the re-verified match
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text_y = y2 + 25
                cv2.putText(frame, f"{check_match} ({check_score:.2f})", (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                continue

            # If we get here, it is GENUINELY a new unknown person
            print(f"New unknown face detected (score: {score:.2f})")
            
            # Generate new name
            new_id = get_next_unknown_id()
            new_name = f"unknown_{new_id}"
            
            # Save current frame to temp file with DESIRED filename
            temp_img_path = f"{new_name}.jpg"
            cv2.imwrite(temp_img_path, face_crop)
            
            try:
                print(f"Registering new person: {new_name}")
                
                # Pass the current embedding to ensure registration succeeds even if crop is bad
                # Use face.normed_embedding which we already have
                new_embeddings = register_face(new_name, temp_img_path, TEMP_DB_ROOT, TEMP_EMB_ROOT, known_embedding=face.normed_embedding)
                
                # Update in-memory database IMMEDIATELY
                db[new_name] = new_embeddings
                
                # Update for this frame
                name = new_name
                
            except Exception as e:
                print(f"Failed to register unknown face: {e}")
                
                # Cleanup failed registration
                try:
                    import shutil
                    failed_folder = os.path.join(TEMP_DB_ROOT, new_name)
                    if os.path.exists(failed_folder):
                        shutil.rmtree(failed_folder)
                    if new_name in db:
                        del db[new_name]
                except Exception as cleanup_e:
                    print(f"Cleanup failed: {cleanup_e}")
            
            # Clean up temp file
            if os.path.exists(temp_img_path):
                os.remove(temp_img_path)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text_y = y2 + 25  # place text below the bounding box
        cv2.putText(frame, f"{name} ({score:.2f})", (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
print(f"Face recognition video saved as {output_path}")
