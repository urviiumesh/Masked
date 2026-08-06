import os
import shutil
import uuid
import numpy as np
from typing import Any

from api.config import FACE_DB_ROOT, EMB_DB_ROOT, TEMP_FACE_DB_ROOT, TEMP_EMB_DB_ROOT, MAX_VIEWS
from register_face import register_face, generate_embeddings, get_app
from tracker_utils import cosine_similarity, recognize_face as match_embedding, face_embedding

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def append_embedding_view(current: np.ndarray, emb: np.ndarray, max_views: int = MAX_VIEWS) -> np.ndarray:
    if current.ndim == 1:
        current = np.expand_dims(current, 0)
    emb = np.asarray(emb, dtype=np.float32).reshape(1, -1)
    if len(current) < max_views:
        return np.vstack([current, emb])
    anchors = max(2, min(len(current), max_views // 5))
    learned_budget = max_views - anchors
    tail = current[anchors:]
    if len(tail) >= learned_budget:
        tail = tail[-(learned_budget - 1):] if learned_budget > 1 else tail[:0]
    if len(tail):
        return np.vstack([current[:anchors], tail, emb])
    return np.vstack([current[:anchors], emb])


def _list_images(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in IMAGE_EXTS and "_occluded" not in f
    )


def _thumbnail_path(name: str, is_unknown: bool = False) -> str | None:
    root = TEMP_FACE_DB_ROOT if is_unknown else FACE_DB_ROOT
    folder = os.path.join(root, name)
    images = _list_images(folder)
    if images:
        return os.path.join(folder, images[0])
    emb_root = TEMP_EMB_DB_ROOT if is_unknown else EMB_DB_ROOT
    npy = os.path.join(emb_root, f"{name}.npy")
    if os.path.isfile(npy):
        return None
    return None


def _person_id(name: str) -> str:
    return str(abs(hash(name)) % 100000).zfill(5)


def list_persons(search: str = "") -> list[dict[str, Any]]:
    persons = []
    for root, emb_root, is_unknown in [
        (FACE_DB_ROOT, EMB_DB_ROOT, False),
        (TEMP_FACE_DB_ROOT, TEMP_EMB_DB_ROOT, True),
    ]:
        if not os.path.isdir(root):
            continue
        for name in sorted(os.listdir(root)):
            folder = os.path.join(root, name)
            if not os.path.isdir(folder):
                continue
            if search and search.lower() not in name.lower():
                continue
            images = _list_images(folder)
            emb_path = os.path.join(emb_root, f"{name}.npy")
            emb_count = 0
            if os.path.isfile(emb_path):
                arr = np.load(emb_path)
                emb_count = int(arr.shape[0]) if arr.ndim > 1 else 1
            persons.append({
                "name": name,
                "id": _person_id(name),
                "is_unknown": is_unknown,
                "image_count": len(images),
                "embedding_count": emb_count,
                "thumbnail": f"/api/persons/{name}/thumbnail" if images else None,
            })
    return persons


def get_person(name: str) -> dict[str, Any] | None:
    for is_unknown in (False, True):
        root = TEMP_FACE_DB_ROOT if is_unknown else FACE_DB_ROOT
        folder = os.path.join(root, name)
        if os.path.isdir(folder):
            images = _list_images(folder)
            return {
                "name": name,
                "id": _person_id(name),
                "is_unknown": is_unknown,
                "images": [f"/api/persons/{name}/images/{img}" for img in images],
                "image_count": len(images),
            }
    return None


def create_person(name: str, image_path: str) -> dict[str, Any]:
    emb = register_face(name, image_path, FACE_DB_ROOT, EMB_DB_ROOT)
    return {"name": name, "embedding_count": int(emb.shape[0]) if emb.ndim > 1 else 1}


def delete_person(name: str) -> bool:
    deleted = False
    for root, emb_root in [(FACE_DB_ROOT, EMB_DB_ROOT), (TEMP_FACE_DB_ROOT, TEMP_EMB_DB_ROOT)]:
        folder = os.path.join(root, name)
        emb = os.path.join(emb_root, f"{name}.npy")
        if os.path.isdir(folder):
            shutil.rmtree(folder)
            deleted = True
        if os.path.isfile(emb):
            os.remove(emb)
            deleted = True
    return deleted


def rename_person(old_name: str, new_name: str) -> bool:
    for root, emb_root in [(FACE_DB_ROOT, EMB_DB_ROOT), (TEMP_FACE_DB_ROOT, TEMP_EMB_DB_ROOT)]:
        old_folder = os.path.join(root, old_name)
        new_folder = os.path.join(root, new_name)
        old_emb = os.path.join(emb_root, f"{old_name}.npy")
        new_emb = os.path.join(emb_root, f"{new_name}.npy")
        if os.path.isdir(old_folder):
            if os.path.exists(new_folder):
                return False
            os.rename(old_folder, new_folder)
        if os.path.isfile(old_emb):
            os.rename(old_emb, new_emb)
    return True


def load_all_embeddings(include_numeric_gallery: bool = False) -> dict[str, np.ndarray]:
    db = {}
    for emb_root in (EMB_DB_ROOT, TEMP_EMB_DB_ROOT):
        if not os.path.isdir(emb_root):
            continue
        for file in os.listdir(emb_root):
            if not file.endswith(".npy"):
                continue
            name = file.replace(".npy", "")
            if name.isdigit() and not include_numeric_gallery:
                continue
            db[name] = np.load(os.path.join(emb_root, file))
    return db


def load_embeddings_for_targets(targets: set[str] | list[str] | None = None) -> dict[str, np.ndarray]:
    if not targets:
        return load_all_embeddings(include_numeric_gallery=False)
    wanted = set(targets)
    db = {}
    for emb_root in (EMB_DB_ROOT, TEMP_EMB_DB_ROOT):
        if not os.path.isdir(emb_root):
            continue
        for name in wanted:
            path = os.path.join(emb_root, f"{name}.npy")
            if os.path.isfile(path):
                db[name] = np.load(path)
    return db


def recognize_embedding(embedding: np.ndarray, db: dict, targets: set[str] | None = None) -> tuple[str, float]:
    if embedding is None:
        return "Unknown", 0.0
    if targets is not None:
        db = {k: v for k, v in db.items() if k in targets}
        if not db:
            return "Unknown", 0.0
    from api.config import THRESHOLD_KNOWN, THRESHOLD_UNKNOWN
    return match_embedding(embedding, db, THRESHOLD_KNOWN, THRESHOLD_UNKNOWN)


def recognize_image(image_path: str, targets: set[str] | None = None) -> list[dict]:
    app = get_app()
    import cv2
    img = cv2.imread(image_path)
    if img is None:
        return []
    db = load_all_embeddings()
    results = []
    for face in app.get(img):
        emb = face_embedding(face)
        name, score = recognize_embedding(emb, db, targets)
        x1, y1, x2, y2 = face.bbox.astype(int)
        results.append({
            "name": name,
            "score": float(score),
            "bbox": [int(x1), int(y1), int(x2), int(y2)],
        })
    return results


def save_upload(file_bytes: bytes, ext: str = ".jpg") -> str:
    from api.config import UPLOAD_DIR
    path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path
