import os
import sys
import numpy as np
import cv2

# --- Setup Paths to FaceRecognitionSystem ---
# Pass this dynamically or assume sibling directory structure
def get_face_sys_paths(base_path=None):
    if base_path is None:
        # Assume we are in GossipNetwork, so go up one level and down to FaceRecognitionSystem
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = os.path.abspath(os.path.join(current_dir, '../FaceRecognitionSystem'))
    
    return {
        "root": base_path,
        "real_db": os.path.join(base_path, "faces_db"),
        "temp_db": os.path.join(base_path, "temp_faces_db"),
        "insightface_pkg": os.path.join(base_path, "insightface_repo", "python-package")
    }

# Ensure InsightFace can be imported
def setup_imports(face_sys_path):
    # Support backward compatibility if a string is passed
    if isinstance(face_sys_path, str):
        paths = [face_sys_path]
    else:
        # Expecting the dict from get_face_sys_paths
        paths = [face_sys_path["root"], face_sys_path.get("insightface_pkg")]

    for p in paths:
        if p and p not in sys.path:
            sys.path.append(p)

# --- Core Logic ---

def setup_model():
    try:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app
    except ImportError as e:
        print(f"Error importing InsightFace: {e}")
        print("Make sure 'insightface' is installed and FaceRecognitionSystem is in path.")
        sys.exit(1)

def load_database(paths):
    db = {}
    real_db = paths["real_db"]
    temp_db = paths["temp_db"]
    
    # helper to read a dir
    def read_dir(directory):
        if not os.path.exists(directory):
            return
        for file in os.listdir(directory):
            if file.endswith(".npy"):
                name = file.replace(".npy", "")
                try:
                    db[name] = np.load(os.path.join(directory, file))
                except Exception as e:
                    print(f"Failed to load {file}: {e}")

    read_dir(real_db)
    read_dir(temp_db)
    print(f"Loaded {len(db)} identities from database.")
    return db

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recognize_face(embedding, db, threshold_known=0.30, threshold_unknown=0.35):
    """
    Returns (name, score) for the best match in db.
    """
    best_match = "Unknown"
    best_score = 0.0
    
    for name, db_emb in db.items():
        if db_emb.ndim == 1:
            score = cosine_similarity(embedding, db_emb)
        else:
            # Multi-view support
            scores = [cosine_similarity(embedding, view) for view in db_emb]
            score = max(scores) if scores else 0.0

        if score > best_score:
            best_score = score
            best_match = name

    # Apply thresholds
    # If the potential match is an "unknown_X", keep strict threshold
    is_unknown_id = best_match.startswith("unknown")
    thresh = threshold_unknown if is_unknown_id else threshold_known
    
    if best_score > thresh:
        return best_match, best_score
    else:
        return "Unknown", best_score


def save_new_face(embedding, paths, db):
    """
    Saves a new unknown face to the temp_db with an incremental ID.
    Returns the new name.
    """
    temp_dir = paths["temp_db"]
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    # Find next available ID
    existing_files = [f for f in os.listdir(temp_dir) if f.startswith("unknown_") and f.endswith(".npy")]
    max_id = 0
    for f in existing_files:
        try:
            # Extract number from "unknown_5.npy"
            parts = f.replace(".npy", "").split("_")
            if len(parts) == 2 and parts[1].isdigit():
                fid = int(parts[1])
                if fid > max_id:
                    max_id = fid
        except:
            continue
            
    new_id = max_id + 1
    new_name = f"unknown_{new_id}"
    filename = os.path.join(temp_dir, f"{new_name}.npy")
    
    try:
        np.save(filename, embedding)
        db[new_name] = embedding
        print(f"Registered new identity: {new_name}")
        return new_name
    except Exception as e:
        print(f"Failed to save new face: {e}")
        return "Unknown"

def get_color(name, target_name, interacted_set):
    """
    Returns BGR color tuple.
    Target: Green
    Interacted: Yellow
    Unknown/Others: Red
    """
    if name == target_name:
        return (0, 255, 0) # Green
    elif name in interacted_set:
        return (0, 255, 255) # Yellow
    else:
        return (0, 0, 255) # Red
