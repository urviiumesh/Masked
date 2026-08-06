import os
import shutil
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Global model instance (Lazy initialization)
_app_instance = None

def get_app() -> FaceAnalysis:
    """Lazy initializer for FaceAnalysis model (CPU mode)."""
    global _app_instance
    if _app_instance is None:
        _app_instance = FaceAnalysis(allowed_modules=['detection', 'recognition'])
        _app_instance.prepare(ctx_id=-1, det_size=(640, 640))
    return _app_instance

# Base directories (Default)
DEFAULT_DB_ROOT = "face_database"   # per‑person image folders
DEFAULT_EMB_ROOT = "faces_db"       # embeddings stored here

def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)

def ensure_person_folder(name: str, db_root: str) -> str:
    """Return the absolute path to the folder for *name*, creating it if needed."""
    folder = os.path.join(db_root, name)
    ensure_dir(folder)
    return folder

def copy_image_to_folder(name: str, image_path: str, db_root: str) -> str:
    """Copy *image_path* into the person's folder.
    If a file with the same name already exists, a numeric suffix is added.
    Returns the final destination path.
    """
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
    """Create a synthetic occlusion (black rectangle over the eyes) and save it.
    The function reads *src_path*, detects the face, draws a rectangle covering the eye region,
    and writes the result to *dest_path*.
    """
    if app is None:
        app = get_app()
    img = cv2.imread(src_path)
    if img is None:
        raise ValueError(f"Unable to read image for augmentation: {src_path}")
    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError("No face detected for augmentation.")
    # Use the first detected face
    face = faces[0]
    # InsightFace returns 5 landmarks: left eye, right eye, nose, left mouth, right mouth
    if not hasattr(face, "kps") or face.kps is None:
        raise ValueError("Landmarks not available for augmentation.")
    landmarks = face.kps  # shape (5, 2)
    # Compute bounding box that covers both eyes
    left_eye = landmarks[0]
    right_eye = landmarks[1]
    # Expand a little to cover the whole eye region
    eye_center = (left_eye + right_eye) / 2
    eye_width = np.linalg.norm(right_eye - left_eye) * 1.5
    eye_height = eye_width * 0.6
    x1 = int(eye_center[0] - eye_width / 2)
    y1 = int(eye_center[1] - eye_height / 2)
    x2 = int(eye_center[0] + eye_width / 2)
    y2 = int(eye_center[1] + eye_height / 2)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.imwrite(dest_path, img)

def augment_mask_image(src_path: str, dest_path: str, app: FaceAnalysis = None) -> None:
    """Create a synthetic lower-face mask occlusion (surgical mask covering nose & mouth) and save it."""
    if app is None:
        app = get_app()
    img = cv2.imread(src_path)
    if img is None:
        raise ValueError(f"Unable to read image for mask augmentation: {src_path}")
    faces = app.get(img)
    if len(faces) == 0:
        raise ValueError("No face detected for mask augmentation.")
    face = faces[0]
    if not hasattr(face, "kps") or face.kps is None:
        raise ValueError("Landmarks not available for mask augmentation.")
    landmarks = face.kps  # shape (5, 2)
    # Keypoints: 0=left_eye, 1=right_eye, 2=nose, 3=left_mouth, 4=right_mouth
    nose = landmarks[2]
    left_mouth = landmarks[3]
    right_mouth = landmarks[4]
    
    # Compute bounding polygon/box for lower face (covering from nose bridge down to chin)
    bbox = face.bbox.astype(int)
    x1, y1, x2, y2 = bbox
    
    mask_y1 = int(nose[1] - (nose[1] - min(landmarks[0][1], landmarks[1][1])) * 0.2)
    mask_y2 = y2
    mask_x1 = max(0, x1)
    mask_x2 = min(img.shape[1], x2)
    
    # Draw dark mask rectangle/polygon over lower face
    cv2.rectangle(img, (mask_x1, mask_y1), (mask_x2, mask_y2), (30, 30, 30), -1)
    cv2.imwrite(dest_path, img)

def generate_embeddings(name: str, db_root: str, emb_root: str, known_embedding: np.ndarray = None, app: FaceAnalysis = None) -> np.ndarray:
    """Compute embeddings for *all* images in the person's folder and store them.
    The resulting .npy file is saved to `emb_root/<name>.npy` with shape (N, 512).
    Returns the generated embeddings.
    """
    if known_embedding is not None:
        # Use provided embedding directly if available
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
        faces = app.get(img)
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
    """Validate the image, copy it, create an occluded version, and update embeddings.
    The occluded image is saved with the suffix `_occluded` before the file extension.
    Returns the generated embeddings.
    """
    if app is None:
        app = get_app()
    # Validate and copy original image
    dest_original = copy_image_to_folder(name, image_path, db_root)
    print(f"Original image copied to {dest_original}")

    # Create eye occluded version (sunglasses)
    base, ext = os.path.splitext(dest_original)
    occluded_path = f"{base}_occluded{ext}"
    try:
        augment_image(dest_original, occluded_path, app=app)
        print(f"Eye occluded image created at {occluded_path}")
    except Exception as e:
        print(f"Warning: could not create eye occluded image – {e}")

    # Create lower-face mask occluded version (surgical mask)
    mask_path = f"{base}_mask{ext}"
    try:
        augment_mask_image(dest_original, mask_path, app=app)
        print(f"Lower-face mask image created at {mask_path}")
    except Exception as e:
        print(f"Warning: could not create lower-face mask image – {e}")

    # Regenerate embeddings for this person (includes original + eye occluded + mask occluded)
    return generate_embeddings(name, db_root, emb_root, known_embedding, app=app)

if __name__ == "__main__":
    person_name = input("Enter name for this person (folder will be created/used): ").strip()
    img_path = input("Enter path to image: ").strip()
    try:
        register_face(person_name, img_path)
    except Exception as e:
        print(f"Error: {e}")
