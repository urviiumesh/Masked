import cv2
import numpy as np
from insightface.app import FaceAnalysis

from register_face import get_video_app, run_face_detection


def _round_down_even(value: int) -> int:
    return max(2, value - (value % 2))


def prepare_detect_frame(frame: np.ndarray, max_width: int = 960) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = 1.0
    if w > max_width:
        scale = max_width / w
        nw = _round_down_even(int(w * scale))
        nh = _round_down_even(int(h * scale))
        frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        scale = nw / w
    else:
        nw = _round_down_even(w)
        nh = _round_down_even(h)
        if nw != w or nh != h:
            frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
            scale = nw / w
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    if not frame.flags["C_CONTIGUOUS"]:
        frame = np.ascontiguousarray(frame)
    return frame, scale


def _is_ort_shape_error(exc: Exception) -> bool:
    msg = str(exc)
    return any(
        token in msg
        for token in ("Dml", "Reshape", "80070057", "RUNTIME_EXCEPTION", "INVALID_ARGUMENT")
    )


def _scale_faces(faces: list, sx: float, sy: float, ox: float = 0.0, oy: float = 0.0) -> list:
    scaled = []
    for face in faces:
        face.bbox = face.bbox.astype(np.float32).copy()
        face.bbox[0] = face.bbox[0] * sx + ox
        face.bbox[2] = face.bbox[2] * sx + ox
        face.bbox[1] = face.bbox[1] * sy + oy
        face.bbox[3] = face.bbox[3] * sy + oy
        if hasattr(face, "kps") and face.kps is not None:
            face.kps = face.kps.astype(np.float32).copy()
            face.kps[:, 0] = face.kps[:, 0] * sx + ox
            face.kps[:, 1] = face.kps[:, 1] * sy + oy
        scaled.append(face)
    return scaled


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def _face_score(face) -> float:
    return float(getattr(face, "det_score", 0.0) or 0.0)


def _nms_faces(faces: list, iou_thresh: float = 0.45) -> list:
    if len(faces) <= 1:
        return faces
    ordered = sorted(faces, key=_face_score, reverse=True)
    kept = []
    for face in ordered:
        if all(_iou(face.bbox, other.bbox) < iou_thresh for other in kept):
            kept.append(face)
    return kept


def _run_detect(app: FaceAnalysis, detect_frame: np.ndarray) -> list:
    try:
        return run_face_detection(app, detect_frame)
    except Exception as e:
        if not _is_ort_shape_error(e):
            raise
        print(f"[DHRISHTI] detect fallback to CPU after ORT error: {e}")
        return run_face_detection(get_video_app(), detect_frame)


def detect_faces(app: FaceAnalysis, frame: np.ndarray, max_width: int = 960) -> list:
    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    detect_frame, scale = prepare_detect_frame(frame, max_width)
    faces = _run_detect(app, detect_frame)
    if scale == 1.0:
        return faces
    inv = 1.0 / scale
    return _scale_faces(faces, inv, inv)


def detect_faces_live(
    app: FaceAnalysis,
    frame: np.ndarray,
    max_width: int = 1280,
    deep: bool = True,
) -> list:
    if frame is None or getattr(frame, "size", 0) == 0:
        return []
    work, scale = prepare_detect_frame(frame, max_width)
    faces = _run_detect(app, work)
    all_faces = list(faces)
    wh, ww = work.shape[:2]
    need_tiles = deep and ww >= 1000
    if not need_tiles and faces:
        smallest = min(max(1.0, float(f.bbox[2] - f.bbox[0])) for f in faces)
        if smallest < 80:
            need_tiles = ww >= 900
    if need_tiles:
        mid = ww // 2
        overlap = max(48, ww // 10)
        tiles = [
            (0, 0, min(ww, mid + overlap), wh),
            (max(0, mid - overlap), 0, ww, wh),
        ]
        for x1, y1, x2, y2 in tiles:
            crop = work[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            tile_faces = _run_detect(app, crop)
            all_faces.extend(_scale_faces(tile_faces, 1.0, 1.0, float(x1), float(y1)))
    merged = _nms_faces(all_faces, iou_thresh=0.4)
    if scale == 1.0:
        return merged
    inv = 1.0 / scale
    return _scale_faces(merged, inv, inv)


def detect_faces_cctv(
    app: FaceAnalysis,
    frame: np.ndarray,
    max_width: int = 1280,
) -> list:
    return detect_faces_live(app, frame, max_width=max_width, deep=True)


def refine_small_face_embedding(app: FaceAnalysis, frame: np.ndarray, bbox, min_side: int = 128):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = int(bw * 0.4)
    pad_y = int(bh * 0.4)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    ch, cw = crop.shape[:2]
    side = min(ch, cw)
    if side < min_side:
        scale = min_side / max(side, 1)
        crop = cv2.resize(
            crop,
            (_round_down_even(int(cw * scale)), _round_down_even(int(ch * scale))),
            interpolation=cv2.INTER_CUBIC,
        )
    faces = detect_faces(app, crop, max_width=max(crop.shape[1], min_side))
    if not faces:
        return None
    best = max(faces, key=_face_score)
    emb = getattr(best, "embedding", None)
    if emb is None:
        emb = getattr(best, "normed_embedding", None)
    return emb
