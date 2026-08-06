import os
import sys
import numpy as np
import cv2


def get_face_sys_paths(base_path=None):
    if base_path is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_path = current_dir
    return {
        "root": base_path,
        "real_db": os.path.join(base_path, "faces_db"),
        "temp_db": os.path.join(base_path, "temp_faces_db"),
        "insightface_pkg": os.path.join(base_path, "insightface_repo", "python-package"),
    }


def setup_imports(face_sys_path):
    if isinstance(face_sys_path, str):
        paths = [face_sys_path]
    else:
        paths = [face_sys_path["root"], face_sys_path.get("insightface_pkg")]
    for p in paths:
        if p and p not in sys.path:
            sys.path.append(p)


def setup_model():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def load_database(paths):
    db = {}
    real_db = paths["real_db"]
    temp_db = paths["temp_db"]

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
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def face_embedding(face):
    emb = getattr(face, "embedding", None)
    if emb is None:
        emb = getattr(face, "normed_embedding", None)
    return emb


def recognize_face(embedding, db, threshold_known=0.35, threshold_unknown=0.35):
    best_match = None
    best_score = 0.0

    for name, db_emb in db.items():
        if db_emb.ndim == 1:
            score = cosine_similarity(embedding, db_emb)
        else:
            scores = [cosine_similarity(embedding, view) for view in db_emb]
            score = max(scores) if scores else 0.0
        if score > best_score:
            best_score = score
            best_match = name

    if best_match is None:
        return "Unknown", 0.0

    is_unknown_id = best_match.startswith("unknown") or best_match.isdigit()
    thresh = threshold_unknown if is_unknown_id else threshold_known
    if best_score >= thresh:
        return best_match, best_score
    return "Unknown", best_score


def save_new_face(embedding, paths, db):
    temp_dir = paths["temp_db"]
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    existing_files = [f for f in os.listdir(temp_dir) if f.startswith("unknown_") and f.endswith(".npy")]
    max_id = 0
    for f in existing_files:
        try:
            parts = f.replace(".npy", "").split("_")
            if len(parts) == 2 and parts[1].isdigit():
                fid = int(parts[1])
                if fid > max_id:
                    max_id = fid
        except Exception:
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
    if name == target_name:
        return (0, 255, 0)
    if name in interacted_set:
        return (0, 255, 255)
    return (0, 0, 255)
