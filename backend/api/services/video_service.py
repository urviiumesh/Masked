import os
import shutil
import subprocess
import threading
import uuid
from typing import Any

import cv2
import numpy as np

from api.config import UPLOAD_DIR, THRESHOLD_KNOWN, THRESHOLD_HIGH_CONF, MAX_VIEWS, EMB_DB_ROOT
from api.services.detect_utils import detect_faces
from api.services.face_service import load_all_embeddings, load_embeddings_for_targets, recognize_embedding
from api.services.log_service import add_log, status_from_score
from register_face import get_video_app
from tracker_utils import face_embedding

_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _detect_max_width() -> int:
    return 1920


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame
    if len(frame.shape) == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    h, w = frame.shape[:2]
    if w % 2 or h % 2:
        frame = cv2.resize(frame, (w - (w % 2), h - (h % 2)), interpolation=cv2.INTER_LINEAR)
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)
    return np.ascontiguousarray(frame)


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(v) for v in value]
    return value


def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _public_job(job: dict) -> dict:
    has_output = bool(job.get("output_path") and os.path.isfile(job.get("output_path", "")))
    input_path = job.get("input_path") or ""
    can_rescan = bool(input_path and os.path.isfile(input_path) and job.get("status") in ("completed", "failed"))
    return _to_json_safe({
        "id": job["id"],
        "status": job["status"],
        "progress": float(job.get("progress", 0)),
        "total_frames": int(job.get("total_frames", 0)),
        "current_frame": int(job.get("current_frame", 0)),
        "filename": job.get("filename", ""),
        "detections": job.get("detections", []),
        "error": job.get("error"),
        "targets": job.get("targets", []),
        "can_rescan": can_rescan,
        "output_url": f"/api/video/jobs/{job['id']}/output" if has_output else None,
    })


def get_job_raw(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def create_job(video_path: str, targets: list[str] | None = None, filename: str = "") -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "progress": 0.0,
            "total_frames": 0,
            "current_frame": 0,
            "output_path": None,
            "input_path": video_path,
            "detections": [],
            "error": None,
            "targets": targets or [],
            "filename": filename or os.path.basename(video_path),
        }
    thread = threading.Thread(target=_process, args=(job_id, video_path, targets), daemon=True)
    thread.start()
    return job_id


def _remove_file(path: str | None):
    if not path:
        return
    try:
        if os.path.isfile(path):
            os.remove(path)
    except OSError:
        pass


def delete_job(job_id: str) -> bool:
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            return False
        if job.get("status") in ("queued", "processing"):
            raise RuntimeError("Cannot delete a job that is still processing")
        input_path = job.get("input_path")
        output_path = job.get("output_path")
        del _jobs[job_id]
    _remove_file(input_path)
    _remove_file(output_path)
    web_path = os.path.join(UPLOAD_DIR, f"output_{job_id}_web.mp4")
    raw_path = os.path.join(UPLOAD_DIR, f"output_{job_id}.mp4")
    _remove_file(web_path)
    _remove_file(raw_path)
    return True


def rescan_job(job_id: str, targets: list[str]) -> dict:
    if not targets:
        raise RuntimeError("Select at least one target before rescanning")
    with _lock:
        job = _jobs.get(job_id)
        if not job:
            raise RuntimeError("Job not found")
        if job.get("status") in ("queued", "processing"):
            raise RuntimeError("Job is already processing")
        input_path = job.get("input_path")
        if not input_path or not os.path.isfile(input_path):
            raise RuntimeError("Original video file is no longer available for rescan")
        old_output = job.get("output_path")
        job["status"] = "queued"
        job["progress"] = 0.0
        job["total_frames"] = 0
        job["current_frame"] = 0
        job["output_path"] = None
        job["detections"] = []
        job["error"] = None
        job["targets"] = list(targets)
    _remove_file(old_output)
    web_path = os.path.join(UPLOAD_DIR, f"output_{job_id}_web.mp4")
    raw_path = os.path.join(UPLOAD_DIR, f"output_{job_id}.mp4")
    _remove_file(web_path)
    _remove_file(raw_path)
    thread = threading.Thread(target=_process, args=(job_id, input_path, targets), daemon=True)
    thread.start()
    return get_job(job_id) or {"id": job_id, "status": "queued"}


def list_jobs() -> list[dict]:
    with _lock:
        jobs = sorted(_jobs.values(), key=lambda j: j["id"], reverse=True)
    return [_public_job(j) for j in jobs]


def get_job(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
    if not job:
        return None
    return _public_job(job)


def _finalize_for_browser(src_path: str, job_id: str) -> str:
    if not os.path.isfile(src_path):
        return src_path
    web_path = os.path.join(UPLOAD_DIR, f"output_{job_id}_web.mp4")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        print("[DHRISHTI] ffmpeg not found — browser may not play mp4v output")
        return src_path
    try:
        result = subprocess.run(
            [
                ffmpeg, "-y", "-i", src_path,
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-preset", "veryfast", "-crf", "23",
                "-an",
                web_path,
            ],
            check=False,
            capture_output=True,
            timeout=600,
        )
        if result.returncode == 0 and os.path.isfile(web_path) and os.path.getsize(web_path) > 0:
            try:
                os.remove(src_path)
            except OSError:
                pass
            return web_path
        err = (result.stderr or b"").decode("utf-8", errors="ignore")[-500:]
        print(f"[DHRISHTI] ffmpeg convert failed ({result.returncode}): {err}")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"[DHRISHTI] ffmpeg convert error: {e}")
    return src_path


def _process(job_id: str, video_path: str, targets: list[str] | None):
    with _lock:
        _jobs[job_id]["status"] = "processing"
    try:
        app = get_video_app()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Cannot open video")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if w <= 0 or h <= 0:
            raise RuntimeError("Invalid video dimensions")
        out_w = w - (w % 2)
        out_h = h - (h % 2)
        out_path = os.path.join(UPLOAD_DIR, f"output_{job_id}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(out_path, fourcc, fps, (out_w, out_h))
        if not out.isOpened():
            raise RuntimeError("Cannot create output video writer")
        db = load_all_embeddings()
        target_set = set(targets or [])
        if not target_set:
            raise RuntimeError("No targets selected")
        db = load_embeddings_for_targets(target_set)
        if not db:
            raise RuntimeError(f"No embeddings found for selected targets: {', '.join(sorted(target_set))}")
        frame_idx = 0
        detections = []
        last_log_at: dict[str, float] = {}
        dirty_faces: set[str] = set()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = _normalize_frame(frame)
            if frame is None or frame.size == 0:
                continue
            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            frame_idx += 1
            timestamp = frame_idx / fps
            frame_boxes: list[dict] = []
            try:
                faces = detect_faces(app, frame, max_width=_detect_max_width())
            except Exception as e:
                print(f"[DHRISHTI] video frame {frame_idx} detect error: {e}")
                faces = []
            for face in faces:
                emb = face_embedding(face)
                if emb is None:
                    continue
                x1, y1, x2, y2 = face.bbox.astype(int)
                name, score = recognize_embedding(emb, db, target_set)
                score_f = float(score)
                if name == "Unknown" or name not in target_set or score_f < THRESHOLD_KNOWN:
                    continue
                frame_boxes.append({
                    "name": name,
                    "score": score_f,
                    "bbox": (int(x1), int(y1), int(x2), int(y2)),
                })
                detections.append({
                    "name": name,
                    "score": score_f,
                    "timestamp": _format_time(timestamp),
                    "frame": int(frame_idx),
                    "start_sec": float(timestamp),
                })
                if score_f >= THRESHOLD_HIGH_CONF and name in db:
                    current = db[name]
                    if current.ndim == 1:
                        current = np.expand_dims(current, 0)
                    updated = np.vstack([current, emb])
                    if len(updated) > MAX_VIEWS:
                        updated = updated[-MAX_VIEWS:]
                    db[name] = updated
                    dirty_faces.add(name)
                last = last_log_at.get(name, -999.0)
                if timestamp - last >= 2.0:
                    last_log_at[name] = timestamp
                    add_log(name, score_f, status_from_score(name, score_f), "Video Processing", source="video")
            for box in frame_boxes:
                x1, y1, x2, y2 = box["bbox"]
                display = box["name"]
                score_f = box["score"]
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{display} ({score_f:.2f})", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            out.write(frame)
            if frame_idx % 5 == 0 or frame_idx == total:
                progress = round(frame_idx / total * 100, 1) if total else 0.0
                with _lock:
                    _jobs[job_id]["current_frame"] = int(frame_idx)
                    _jobs[job_id]["total_frames"] = int(total)
                    _jobs[job_id]["progress"] = float(progress)
                    _jobs[job_id]["detections"] = detections[-200:]
        cap.release()
        out.release()
        for name in dirty_faces:
            try:
                np.save(os.path.join(EMB_DB_ROOT, f"{name}.npy"), db[name])
            except Exception as e:
                print(f"[DHRISHTI] failed saving embeddings for {name}: {e}")
        final_path = _finalize_for_browser(out_path, job_id)
        with _lock:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["output_path"] = final_path
            _jobs[job_id]["detections"] = detections
            _jobs[job_id]["progress"] = 100.0
            _jobs[job_id]["current_frame"] = int(frame_idx)
            _jobs[job_id]["total_frames"] = int(total or frame_idx)
    except Exception as e:
        with _lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)
