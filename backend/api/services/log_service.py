import json
import os
import threading
import uuid
from datetime import datetime
from typing import Any

import numpy as np

from api.config import LOGS_DIR, SNAPSHOTS_DIR

LOG_FILE = os.path.join(LOGS_DIR, "events.json")
_log_lock = threading.RLock()


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


def _load() -> list[dict]:
    if not os.path.isfile(LOG_FILE):
        return []
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, ValueError):
        return []


def _save(events: list[dict]) -> None:
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(_to_json_safe(events), f, indent=2)


def add_log(
    name: str,
    score: float,
    status: str,
    location: str = "Live Feed",
    snapshot_path: str | None = None,
    source: str = "livestream",
) -> dict[str, Any]:
    with _log_lock:
        events = _load()
        entry = {
            "id": uuid.uuid4().hex[:8].upper(),
            "name": name,
            "score": round(float(score), 4),
            "status": status,
            "location": location,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "snapshot_url": f"/api/logs/snapshots/{os.path.basename(snapshot_path)}" if snapshot_path else None,
            "source": source,
        }
        events.insert(0, entry)
        events = events[:500]
        _save(events)
    return entry


def get_logs(limit: int = 50, source: str | None = None) -> list[dict]:
    with _log_lock:
        events = _load()
        if source:
            events = [e for e in events if e.get("source") == source]
        return events[:limit]


def clear_logs(source: str | None = None) -> dict[str, int]:
    """Delete stored events and the snapshots belonging to them."""
    with _log_lock:
        events = _load()
        removed = events if source is None else [e for e in events if e.get("source") == source]
        retained = [] if source is None else [e for e in events if e.get("source") != source]

        snapshot_names = {
            os.path.basename(str(event["snapshot_url"]))
            for event in removed
            if event.get("snapshot_url")
        }
        deleted_snapshots = 0
        for name in snapshot_names:
            path = os.path.join(SNAPSHOTS_DIR, name)
            if os.path.isfile(path):
                os.remove(path)
                deleted_snapshots += 1

        if retained:
            _save(retained)
        elif os.path.isfile(LOG_FILE):
            os.remove(LOG_FILE)

        # Exports may still contain the deleted events, so discard the cached file.
        export_path = os.path.join(LOGS_DIR, "export.xlsx")
        if os.path.isfile(export_path):
            os.remove(export_path)

        return {"deleted": len(removed), "deleted_snapshots": deleted_snapshots}


def save_snapshot(frame_bytes: bytes) -> str:
    name = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(SNAPSHOTS_DIR, name)
    with open(path, "wb") as f:
        f.write(frame_bytes)
    return path


def export_xlsx(source: str | None = None) -> str:
    from openpyxl import Workbook
    events = get_logs(limit=500, source=source)
    wb = Workbook()
    ws = wb.active
    ws.title = "Detection Logs"
    ws.append(["ID", "Name", "Confidence", "Status", "Location", "Timestamp", "Source"])
    for e in events:
        ws.append([e["id"], e["name"], e["score"], e["status"], e["location"], e["timestamp"], e.get("source", "")])
    out = os.path.join(LOGS_DIR, "export.xlsx")
    wb.save(out)
    return out


def status_from_score(name: str, score: float) -> str:
    if name == "Unknown":
        return "ALERT"
    if score >= 0.55:
        return "MATCH"
    if score >= 0.40:
        return "PARTIAL"
    return "TRACK"
