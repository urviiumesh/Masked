import json
import os

import numpy as np
import pytest

from api.services import log_service


@pytest.fixture
def isolated_logs(tmp_path, monkeypatch):
    logs_dir = tmp_path / "logs"
    snaps_dir = tmp_path / "snaps"
    logs_dir.mkdir()
    snaps_dir.mkdir()
    monkeypatch.setattr(log_service, "LOGS_DIR", str(logs_dir))
    monkeypatch.setattr(log_service, "SNAPSHOTS_DIR", str(snaps_dir))
    monkeypatch.setattr(log_service, "LOG_FILE", str(logs_dir / "events.json"))
    return logs_dir, snaps_dir


def test_status_from_score():
    assert log_service.status_from_score("Unknown", 0.9) == "ALERT"
    assert log_service.status_from_score("Alice", 0.6) == "MATCH"
    assert log_service.status_from_score("Alice", 0.45) == "PARTIAL"
    assert log_service.status_from_score("Alice", 0.2) == "TRACK"


def test_add_and_get_logs(isolated_logs):
    entry = log_service.add_log("Alice", 0.77, "MATCH", location="Cam1", source="livestream")
    assert entry["name"] == "Alice"
    assert entry["score"] == 0.77
    assert entry["id"]
    logs = log_service.get_logs(limit=10)
    assert len(logs) == 1
    assert logs[0]["name"] == "Alice"


def test_get_logs_filters_source(isolated_logs):
    log_service.add_log("A", 0.5, "MATCH", source="livestream")
    log_service.add_log("B", 0.5, "MATCH", source="video")
    assert len(log_service.get_logs(source="video")) == 1
    assert log_service.get_logs(source="video")[0]["name"] == "B"


def test_save_snapshot(isolated_logs):
    _, snaps = isolated_logs
    path = log_service.save_snapshot(b"fakejpeg")
    assert os.path.isfile(path)
    assert path.startswith(str(snaps))


def test_export_xlsx(isolated_logs):
    log_service.add_log("Alice", 0.8, "MATCH", source="livestream")
    path = log_service.export_xlsx()
    assert os.path.isfile(path)
    assert path.endswith(".xlsx")


def test_clear_logs_removes_only_requested_source_and_its_snapshot(isolated_logs):
    _, snaps = isolated_logs
    snapshot = snaps / "live.jpg"
    snapshot.write_bytes(b"fakejpeg")
    log_service.add_log("Live", 0.8, "MATCH", snapshot_path=str(snapshot), source="livestream")
    log_service.add_log("Video", 0.7, "MATCH", source="video")

    result = log_service.clear_logs("livestream")

    assert result == {"deleted": 1, "deleted_snapshots": 1}
    assert not snapshot.exists()
    assert [event["name"] for event in log_service.get_logs()] == ["Video"]


def test_clear_logs_removes_log_file_and_cached_export(isolated_logs):
    logs, _ = isolated_logs
    log_service.add_log("Live", 0.8, "MATCH")
    export = log_service.export_xlsx()

    result = log_service.clear_logs()

    assert result["deleted"] == 1
    assert not os.path.exists(log_service.LOG_FILE)
    assert not os.path.exists(export)
    assert not list(logs.iterdir())


def test_to_json_safe_numpy():
    payload = {"score": np.float32(0.5), "bbox": np.array([1, 2, 3])}
    safe = log_service._to_json_safe(payload)
    assert isinstance(safe["score"], float)
    assert safe["bbox"] == [1, 2, 3]
    json.dumps(safe)
