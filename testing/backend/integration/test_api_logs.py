import json
import os

import pytest


@pytest.fixture
def isolated_api_logs(tmp_path, monkeypatch):
    from api.services import log_service

    logs_dir = tmp_path / "logs"
    snaps_dir = tmp_path / "snaps"
    logs_dir.mkdir()
    snaps_dir.mkdir()
    monkeypatch.setattr(log_service, "LOGS_DIR", str(logs_dir))
    monkeypatch.setattr(log_service, "SNAPSHOTS_DIR", str(snaps_dir))
    monkeypatch.setattr(log_service, "LOG_FILE", str(logs_dir / "events.json"))
    log_service.add_log("Tester", 0.91, "MATCH", location="Unit", source="livestream")
    log_service.add_log("Clip", 0.5, "PARTIAL", location="File", source="video")
    return logs_dir


def test_get_logs(client, isolated_api_logs):
    res = client.get("/api/logs?limit=10")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body, list)
    assert len(body) >= 2


def test_get_logs_source_filter(client, isolated_api_logs):
    res = client.get("/api/logs?source=video")
    assert res.status_code == 200
    body = res.json()
    assert all(e.get("source") == "video" for e in body)
    assert any(e["name"] == "Clip" for e in body)


def test_export_logs(client, isolated_api_logs):
    res = client.get("/api/logs/export")
    assert res.status_code == 200
    assert (
        "spreadsheet" in res.headers.get("content-type", "")
        or "octet-stream" in res.headers.get("content-type", "")
        or res.headers.get("content-type", "").startswith("application/")
    )
    assert len(res.content) > 0


def test_snapshot_missing(client, isolated_api_logs):
    res = client.get("/api/logs/snapshots/does-not-exist.jpg")
    assert res.status_code == 404
