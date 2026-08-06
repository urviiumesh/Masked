import asyncio
import os
import queue
import sys
import threading
import time
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from api.config import THRESHOLD_HIGH_CONF, MAX_VIEWS, EMB_DB_ROOT, TEMP_EMB_DB_ROOT
from api.services.camera_presets import get_preset, preset_to_rtsp_urls
from api.services.detect_utils import detect_faces_live, refine_small_face_embedding
from api.services.face_service import load_all_embeddings, load_embeddings_for_targets, recognize_embedding
from api.services.log_service import add_log, status_from_score, save_snapshot
from api.services.runtime_device import active_provider, runtime_info
from register_face import get_stream_app, reset_stream_app
from tracker_utils import face_embedding

STALL_RECONNECT_SEC = 8.0
READ_FAIL_LIMIT = 60
RTSP_FLUSH_MAX = 4
OVERLAY_TTL_SEC = 5.0
JPEG_QUALITY = 55
DISPLAY_MAX_WIDTH = 960
DETECT_SOURCE_MAX_WIDTH = 1280
LOG_COOLDOWN_SEC = 1.5
SMALL_FACE_PX = 100
PUBLISH_MIN_INTERVAL = 1.0 / 12.0
STREAM_MATCH_FLOOR = 0.35


class StreamManager:
    def __init__(self):
        self._lock = threading.Lock()
        self._cap = None
        self._running = False
        self._source = ""
        self._connect_sources: list[str] = []
        self._latest_jpeg: bytes | None = None
        self._frame_seq = 0
        self._fps = 0.0
        self._detect_fps = 0.0
        self._width = 0
        self._height = 0
        self._display_width = 0
        self._display_height = 0
        self._capture_thread: threading.Thread | None = None
        self._detect_thread: threading.Thread | None = None
        self._detect_queue: queue.Queue = queue.Queue(maxsize=1)
        self._overlay_faces: list[dict] = []
        self._overlay_lock = threading.Lock()
        self._db: dict = {}
        self._dirty: set[str] = set()
        self._active_targets: set[str] = set()
        self._location = "Live Feed"
        self._subscribers: list = []
        self._frame_subscribers: list = []
        self._last_log_time: dict[str, float] = {}
        self._app = None
        self._last_good_frame_at = 0.0
        self._read_fail_streak = 0
        self._detect_times: list[float] = []
        self._dropped_frames = 0
        self._last_face_count = 0
        self._last_match_count = 0
        self._last_publish_at = 0.0

    def _get_app(self):
        if self._app is None:
            self._app = get_stream_app()
        return self._app

    def _open_capture(self, source: str):
        if source.startswith("rtsp://") or source.startswith("http://") or source.startswith("https://"):
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                "rtsp_transport;tcp|stimeout;5000000|max_delay;0|fflags;nobuffer|flags;low_delay"
            )
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass
            return cap
        if source.startswith("webcam:"):
            idx = int(source.split(":")[1])
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else 0
            return cv2.VideoCapture(idx, backend)
        return cv2.VideoCapture(source)

    def _try_open(self, sources: list[str]):
        for src in sources:
            cap = self._open_capture(src)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    return cap, src
                cap.release()
        raise RuntimeError(f"Could not open stream. Tried: {', '.join(sources)}")

    def _read_latest_frame(self):
        if not self._cap:
            return False, None
        is_network = self._source.startswith("rtsp://") or self._source.startswith("http")
        if not is_network:
            return self._cap.read()
        grabbed = 0
        while grabbed < RTSP_FLUSH_MAX:
            if not self._cap.grab():
                break
            grabbed += 1
        if grabbed == 0:
            return False, None
        if grabbed > 1:
            self._dropped_frames += grabbed - 1
        return self._cap.retrieve()

    def _reconnect_capture(self) -> bool:
        if not self._connect_sources:
            return False
        if self._cap:
            self._cap.release()
            self._cap = None
        try:
            cap, opened = self._try_open(self._connect_sources)
            self._cap = cap
            self._source = opened
            self._read_fail_streak = 0
            self._last_good_frame_at = time.time()
            return True
        except Exception:
            return False

    def _downscale(self, frame: np.ndarray, max_width: int) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= max_width:
            if not frame.flags["C_CONTIGUOUS"]:
                return np.ascontiguousarray(frame)
            return frame
        scale = max_width / w
        nw = max(2, int(w * scale) & ~1)
        nh = max(2, int(h * scale) & ~1)
        return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

    def _prune_overlays(self, now: float):
        self._overlay_faces = [o for o in self._overlay_faces if o["expires_at"] > now]

    def _merge_overlays(self, new_items: list[dict], now: float):
        self._prune_overlays(now)
        by_key: dict[str, dict] = {}
        for o in self._overlay_faces:
            by_key[o.get("track_key", o["name"])] = o
        for item in new_items:
            by_key[item.get("track_key", item["name"])] = item
        self._overlay_faces = list(by_key.values())

    def _draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        now = time.time()
        with self._overlay_lock:
            self._prune_overlays(now)
            overlays = list(self._overlay_faces)
        if not overlays:
            return frame
        annotated = frame
        for item in overlays:
            x1, y1, x2, y2 = item["bbox"]
            x1 = max(0, min(int(x1), annotated.shape[1] - 1))
            y1 = max(0, min(int(y1), annotated.shape[0] - 1))
            x2 = max(0, min(int(x2), annotated.shape[1] - 1))
            y2 = max(0, min(int(y2), annotated.shape[0] - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = item["label"]
            stamp = item.get("timestamp", "")
            text = f"{label}  {stamp}" if stamp else label
            text_y = max(y1 - 8, 16)
            cv2.putText(
                annotated,
                text,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                3,
            )
            cv2.putText(
                annotated,
                text,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
            )
        return annotated

    def _publish_frame(self, annotated: np.ndarray):
        ok, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        jpeg_bytes = jpeg.tobytes() if ok else None
        with self._lock:
            self._latest_jpeg = jpeg_bytes
            self._frame_seq += 1
            seq = self._frame_seq
        if jpeg_bytes:
            for q in list(self._frame_subscribers):
                try:
                    while not q.empty():
                        q.get_nowait()
                    q.put_nowait({"type": "frame", "seq": seq, "jpeg": jpeg_bytes})
                except Exception:
                    pass

    def connect(self, source: str, location: str = "Live Feed", targets: list[str] | None = None, preset_id: str | None = None) -> dict:
        self.disconnect()
        sources = [source] if source else []
        if preset_id:
            preset = get_preset(preset_id)
            if not preset:
                raise RuntimeError(f"Unknown camera preset: {preset_id}")
            sources = preset_to_rtsp_urls(preset)
            location = preset.get("location", location)
            source = sources[0]
        self._connect_sources = sources
        cap, opened_source = self._try_open(sources)
        self._cap = cap
        self._source = opened_source
        self._location = location
        self._db = load_all_embeddings()
        if targets:
            self._active_targets = {t for t in targets if t in self._db or os.path.isfile(os.path.join(EMB_DB_ROOT, f"{t}.npy"))}
            missing = [t for t in targets if t not in self._active_targets]
            if missing:
                print(f"[DHRISHTI] stream targets missing embeddings: {missing}")
            if self._active_targets:
                self._db = load_embeddings_for_targets(self._active_targets)
        else:
            self._active_targets = set(self._db.keys())
        self._width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        self._height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        self._display_width = min(self._width, DISPLAY_MAX_WIDTH)
        self._display_height = int(self._height * (self._display_width / max(self._width, 1)))
        self._frame_seq = 0
        self._read_fail_streak = 0
        self._dropped_frames = 0
        self._last_good_frame_at = time.time()
        self._overlay_faces = []
        self._latest_jpeg = None
        self._detect_times = []
        self._last_face_count = 0
        self._last_match_count = 0
        self._last_log_time = {}
        self._detect_queue = queue.Queue(maxsize=1)
        self._app = None
        reset_stream_app()
        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
        self._capture_thread.start()
        self._detect_thread.start()
        print(
            f"[DHRISHTI] stream connected source={self._source} "
            f"targets={sorted(self._active_targets)} identities={len(self._db)} "
            f"provider={active_provider()} cctv_detect_w={DETECT_SOURCE_MAX_WIDTH}"
        )
        return self.status()

    def disconnect(self):
        self._running = False
        for thread in (self._capture_thread, self._detect_thread):
            if thread and thread.is_alive():
                thread.join(timeout=3)
        if self._cap:
            self._cap.release()
            self._cap = None
        self._connect_sources = []
        self._save_dirty()
        self._capture_thread = None
        self._detect_thread = None
        self._latest_jpeg = None
        self._frame_seq = 0
        self._overlay_faces = []

    def _save_dirty(self):
        for name in self._dirty:
            if name in self._db:
                root = TEMP_EMB_DB_ROOT if name.startswith("unknown") else EMB_DB_ROOT
                os.makedirs(root, exist_ok=True)
                np.save(os.path.join(root, f"{name}.npy"), self._db[name])
        self._dirty.clear()

    def _capture_loop(self):
        frame_times: list[float] = []
        while self._running and self._cap and self._cap.isOpened():
            ret, frame = self._read_latest_frame()
            now = time.time()
            if not ret or frame is None:
                self._read_fail_streak += 1
                stalled = now - self._last_good_frame_at > STALL_RECONNECT_SEC
                if stalled or self._read_fail_streak >= READ_FAIL_LIMIT:
                    if not self._reconnect_capture():
                        time.sleep(0.05)
                    continue
                time.sleep(0.001)
                continue
            self._read_fail_streak = 0
            self._last_good_frame_at = now
            display = self._downscale(frame, DISPLAY_MAX_WIDTH)
            self._display_width = display.shape[1]
            self._display_height = display.shape[0]
            if now - self._last_publish_at >= PUBLISH_MIN_INTERVAL:
                annotated = self._draw_overlays(display)
                self._publish_frame(annotated)
                self._last_publish_at = now
                frame_times.append(now)
                frame_times = [t for t in frame_times if now - t < 1.0]
                self._fps = len(frame_times)
            detect_frame = self._downscale(frame, DETECT_SOURCE_MAX_WIDTH)
            sx = self._display_width / max(detect_frame.shape[1], 1)
            sy = self._display_height / max(detect_frame.shape[0], 1)
            try:
                while True:
                    self._detect_queue.get_nowait()
                    self._dropped_frames += 1
            except queue.Empty:
                pass
            try:
                self._detect_queue.put_nowait({
                    "frame": detect_frame,
                    "scale_x": sx,
                    "scale_y": sy,
                })
            except queue.Full:
                self._dropped_frames += 1

    def _detect_loop(self):
        app = self._get_app()
        while self._running:
            try:
                item = self._detect_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            if not self._running:
                break
            if not self._active_targets:
                with self._overlay_lock:
                    self._overlay_faces = []
                self._last_face_count = 0
                self._last_match_count = 0
                continue
            frame = item["frame"]
            scale_x = float(item["scale_x"])
            scale_y = float(item["scale_y"])
            now = time.time()
            stamp = datetime.now().strftime("%H:%M:%S")
            new_overlays = []
            try:
                faces = detect_faces_live(app, frame, max_width=DETECT_SOURCE_MAX_WIDTH, deep=True)
            except Exception as e:
                print(f"[DHRISHTI] detect error: {e}")
                continue
            self._last_face_count = len(faces)
            match_count = 0
            for face in faces:
                emb = face_embedding(face)
                if emb is None:
                    continue
                x1, y1, x2, y2 = face.bbox.astype(int)
                face_w = max(1, x2 - x1)
                face_h = max(1, y2 - y1)
                name, score = recognize_embedding(emb, self._db, self._active_targets)
                score_f = float(score)
                if min(face_w, face_h) < SMALL_FACE_PX:
                    refined = refine_small_face_embedding(app, frame, (x1, y1, x2, y2))
                    if refined is not None:
                        rname, rscore = recognize_embedding(refined, self._db, self._active_targets)
                        if float(rscore) >= score_f:
                            name, score_f, emb = rname, float(rscore), refined
                if name == "Unknown" or name not in self._active_targets or score_f < STREAM_MATCH_FLOOR:
                    continue
                match_count += 1
                dx1 = int(x1 * scale_x)
                dy1 = int(y1 * scale_y)
                dx2 = int(x2 * scale_x)
                dy2 = int(y2 * scale_y)
                label = f"{name} ({score_f:.2f})"
                new_overlays.append({
                    "name": name,
                    "track_key": name,
                    "bbox": (dx1, dy1, dx2, dy2),
                    "label": label,
                    "timestamp": stamp,
                    "expires_at": now + OVERLAY_TTL_SEC,
                })
                if score_f >= THRESHOLD_HIGH_CONF and name in self._db and min(face_w, face_h) >= SMALL_FACE_PX:
                    current = self._db[name]
                    if current.ndim == 1:
                        current = np.expand_dims(current, 0)
                    updated = np.vstack([current, emb])
                    if len(updated) > MAX_VIEWS:
                        updated = updated[-MAX_VIEWS:]
                    self._db[name] = updated
                    self._dirty.add(name)
                last = self._last_log_time.get(name, 0)
                if now - last >= LOG_COOLDOWN_SEC:
                    self._last_log_time[name] = now
                    status = status_from_score(name, score_f)
                    snap = None
                    try:
                        pad = max(12, int(min(face_w, face_h) * 0.2))
                        h, w = frame.shape[:2]
                        cx1 = max(0, x1 - pad)
                        cy1 = max(0, y1 - pad)
                        cx2 = min(w, x2 + pad)
                        cy2 = min(h, y2 + pad)
                        crop = frame[cy1:cy2, cx1:cx2]
                        if crop.size > 0:
                            tagged = crop.copy()
                            if min(tagged.shape[:2]) < 96:
                                scale = 96 / max(1, min(tagged.shape[:2]))
                                tagged = cv2.resize(
                                    tagged,
                                    (max(2, int(tagged.shape[1] * scale)), max(2, int(tagged.shape[0] * scale))),
                                    interpolation=cv2.INTER_CUBIC,
                                )
                            cv2.putText(
                                tagged,
                                f"{name} {stamp}",
                                (8, 20),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                            )
                            _, buf = cv2.imencode(".jpg", tagged, [cv2.IMWRITE_JPEG_QUALITY, 85])
                            snap = save_snapshot(buf.tobytes())
                    except Exception:
                        pass
                    entry = add_log(name, score_f, status, self._location, snap, source="livestream")
                    for q in list(self._subscribers):
                        try:
                            q.put_nowait(entry)
                        except Exception:
                            pass
            self._last_match_count = match_count
            self._detect_times.append(time.time())
            self._detect_times = [t for t in self._detect_times if self._detect_times[-1] - t < 1.0]
            self._detect_fps = len(self._detect_times)
            with self._overlay_lock:
                self._merge_overlays(new_overlays, now)

    def get_frame(self) -> tuple[np.ndarray | None, int]:
        return None, self._frame_seq

    def get_jpeg(self) -> tuple[bytes | None, int]:
        with self._lock:
            return self._latest_jpeg, self._frame_seq

    def status(self) -> dict[str, Any]:
        info = runtime_info()
        stream_provider = "CPUExecutionProvider"
        if active_provider() not in ("DmlExecutionProvider", "CPUExecutionProvider"):
            stream_provider = info["active_provider"]
        return {
            "connected": self._running,
            "source": self._source,
            "fps": round(self._fps, 1),
            "detect_fps": round(self._detect_fps, 1),
            "resolution": f"{self._width}x{self._height}",
            "display_resolution": f"{self._display_width}x{self._display_height}",
            "dropped_frames": self._dropped_frames,
            "faces_seen": self._last_face_count,
            "matches": self._last_match_count,
            "location": self._location,
            "active_targets": list(self._active_targets),
            "identity_count": len(self._db),
            "frame_seq": self._frame_seq,
            "gpu_enabled": info["gpu_enabled"] and stream_provider != "CPUExecutionProvider",
            "ort_provider": stream_provider,
        }

    def set_targets(self, targets: list[str]):
        valid = set()
        for t in targets:
            if t in self._db or os.path.isfile(os.path.join(EMB_DB_ROOT, f"{t}.npy")):
                valid.add(t)
        self._active_targets = valid
        if valid:
            self._db.update(load_embeddings_for_targets(valid))
        print(f"[DHRISHTI] stream targets updated: {sorted(valid)}")

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._subscribers.append(q)
        return q

    def unsubscribe(self, q: asyncio.Queue):
        if q in self._subscribers:
            self._subscribers.remove(q)

    def subscribe_frames(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._frame_subscribers.append(q)
        return q

    def unsubscribe_frames(self, q: asyncio.Queue):
        if q in self._frame_subscribers:
            self._frame_subscribers.remove(q)


stream_manager = StreamManager()
