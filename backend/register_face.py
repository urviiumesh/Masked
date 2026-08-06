import os
import shutil
import threading

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from api.services.runtime_device import (
    active_provider,
    ctx_id_for_provider,
    face_analysis_kwargs,
    runtime_info,
)

_app_instance = None
_app_stream_instance = None
_app_video_instance = None
_runtime_logged = False
_inference_lock = threading.Lock()
_stream_inference_lock = threading.Lock()
_video_inference_lock = threading.Lock()
_init_lock = threading.Lock()


def inference_lock() -> threading.Lock:
    return _inference_lock


def _log_runtime_once():
    global _runtime_logged
    if _runtime_logged:
        return
    info = runtime_info()
    print(
        f"[DHRISHTI] ORT provider={info['active_provider']} "
        f"gpu={info['gpu_enabled']} available={info['available_providers']}"
    )
    if not info["gpu_enabled"]:
        print(f"[DHRISHTI] GPU hint: {info['hint']}")
    _runtime_logged = True


def _create_app(
    det_size: tuple[int, int],
    providers: list[str] | None = None,
    ctx_id: int | None = None,
    det_thresh: float = 0.45,
) -> FaceAnalysis:
    _log_runtime_once()
    if providers is None:
        kwargs = face_analysis_kwargs()
        cid = ctx_id_for_provider() if ctx_id is None else ctx_id
    else:
        kwargs = {"providers": providers}
        cid = -1 if providers == ["CPUExecutionProvider"] else (ctx_id if ctx_id is not None else ctx_id_for_provider())
    app = FaceAnalysis(allowed_modules=["detection", "recognition"], **kwargs)
    app.prepare(ctx_id=cid, det_thresh=det_thresh, det_size=det_size)
    return app


def stream_det_size() -> tuple[int, int]:
    return (960, 960)


def video_det_size() -> tuple[int, int]:
    return (640, 640)


def get_app() -> FaceAnalysis:
    return get_video_app()


def get_video_app() -> FaceAnalysis:
    global _app_video_instance
    with _init_lock:
        if _app_video_instance is None:
            provider = active_provider()
            if provider == "CUDAExecutionProvider":
                print("[DHRISHTI] Video processing using CUDAExecutionProvider")
                _app_video_instance = _create_app(video_det_size(), det_thresh=0.5)
            else:
                print("[DHRISHTI] Video processing using CPUExecutionProvider (Intel DirectML skipped — unsafe for SCRFD)")
                _app_video_instance = _create_app(
                    video_det_size(),
                    providers=["CPUExecutionProvider"],
                    ctx_id=-1,
                    det_thresh=0.5,
                )
        return _app_video_instance


def get_stream_app() -> FaceAnalysis:
    global _app_stream_instance
    with _init_lock:
        if _app_stream_instance is None:
            provider = active_provider()
            if provider == "CUDAExecutionProvider":
                print("[DHRISHTI] Stream CCTV detection using CUDAExecutionProvider det=960 thresh=0.35")
                _app_stream_instance = _create_app(stream_det_size(), det_thresh=0.35)
            else:
                print("[DHRISHTI] Stream CCTV detection using CPUExecutionProvider det=960 thresh=0.35")
                _app_stream_instance = _create_app(
                    stream_det_size(),
                    providers=["CPUExecutionProvider"],
                    ctx_id=-1,
                    det_thresh=0.35,
                )
        return _app_stream_instance


def reset_stream_app():
    global _app_stream_instance
    with _init_lock:
        _app_stream_instance = None


def run_face_detection(app: FaceAnalysis, frame: np.ndarray):
    if frame is None or frame.size == 0:
        return []
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)
    lock = _stream_inference_lock if app is _app_stream_instance else _video_inference_lock
    with lock:
        return app.get(frame)


DEFAULT_DB_ROOT = "face_database"
DEFAULT_EMB_ROOT = "faces_db"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def ensure_person_folder(name: str, db_root: str) -> str:
    folder = os.path.join(db_root, name)
    ensure_dir(folder)
    return folder


def copy_image_to_folder(name: str, image_path: str, db_root: str) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    dest_folder = ensure_person_folder(name, db_root)
    base_name = os.path.basename(image_path)
    dest_path = os.path.join(dest_folder, base_name)
    if os.path.exists(dest_path):
        name_root, ext = os.path.splitext(base_name)
        counter = 1
        while True:
            new_name = f"{name_root}_{counter}{ext}"
            dest_path = os.path.join(dest_folder, new_name)
            if not os.path.exists(dest_path):
                break
            counter += 1
    shutil.copy2(image_path, dest_path)
    return dest_path


def augment_image(src_path: str, dest_path: str, app: FaceAnalysis = None) -> None:
    if app is None:
        app = get_app()
    img = cv2.imread(src_path)
    if img is None:
        raise ValueError(f"Unable to read image for augmentation: {src_path}")
    faces = run_face_detection(app, img)
    if len(faces) == 0:
        raise ValueError("No face detected for augmentation.")
    face = faces[0]
    if not hasattr(face, "kps") or face.kps is None:
        raise ValueError("Landmarks not available for augmentation.")
    landmarks = face.kps
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    eye_center = (left_eye + right_eye) / 2
    eye_width = np.linalg.norm(right_eye - left_eye) * 1.5
    eye_height = eye_width * 0.6
    x1 = int(eye_center[0] - eye_width / 2)
    y1 = int(eye_center[1] - eye_height / 2)
    x2 = int(eye_center[0] + eye_width / 2)
    y2 = int(eye_center[1] + eye_height / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.imwrite(dest_path, img)


def generate_embeddings(name: str, db_root: str, emb_root: str, known_embedding: np.ndarray = None, app: FaceAnalysis = None) -> np.ndarray:
    if known_embedding is not None:
        embeddings = [known_embedding]
        ensure_dir(emb_root)
        emb_array = np.stack(embeddings)
        np.save(os.path.join(emb_root, f"{name}.npy"), emb_array)
        print(f"Saved {len(embeddings)} embeddings for '{name}' to {emb_root}/{name}.npy")
        return emb_array

    if app is None:
        app = get_app()
    person_folder = os.path.join(db_root, name)
    if not os.path.isdir(person_folder):
        raise FileNotFoundError(f"Person folder not found: {person_folder}")
    embeddings = []
    for fname in os.listdir(person_folder):
        img_path = os.path.join(person_folder, fname)
        if not os.path.isfile(img_path):
            continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        faces = run_face_detection(app, img)
        if len(faces) == 0:
            continue
        embeddings.append(faces[0].embedding)
    if not embeddings:
        raise RuntimeError(f"No valid faces found for person '{name}'.")
    ensure_dir(emb_root)
    emb_array = np.stack(embeddings)
    np.save(os.path.join(emb_root, f"{name}.npy"), emb_array)
    print(f"Saved {len(embeddings)} embeddings for '{name}' to {emb_root}/{name}.npy")
    return emb_array


def register_face(name: str, image_path: str, db_root: str = DEFAULT_DB_ROOT, emb_root: str = DEFAULT_EMB_ROOT, known_embedding: np.ndarray = None, app: FaceAnalysis = None) -> np.ndarray:
    if app is None:
        app = get_app()
    dest_original = copy_image_to_folder(name, image_path, db_root)
    print(f"Original image copied to {dest_original}")

    base, ext = os.path.splitext(dest_original)
    occluded_path = f"{base}_occluded{ext}"
    try:
        augment_image(dest_original, occluded_path, app=app)
        print(f"Occluded image created at {occluded_path}")
    except Exception as e:
        print(f"Warning: could not create occluded image – {e}")

    return generate_embeddings(name, db_root, emb_root, known_embedding, app=app)


if __name__ == "__main__":
    person_name = input("Enter name for this person (folder will be created/used): ").strip()
    img_path = input("Enter path to image: ").strip()
    try:
        register_face(person_name, img_path)
    except Exception as e:
        print(f"Error: {e}")
