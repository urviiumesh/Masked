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


def test_clear_logs_by_source_deletes_events_and_snapshots(client, isolated_api_logs):
    from api.services import log_service

    snapshot = os.path.join(log_service.SNAPSHOTS_DIR, "livestream.jpg")
    with open(snapshot, "wb") as f:
        f.write(b"snapshot")
    log_service.add_log(
        "With snapshot",
        0.88,
        "MATCH",
        snapshot_path=snapshot,
        source="livestream",
    )

    res = client.delete("/api/logs?source=livestream")

    assert res.status_code == 200
    assert res.json() == {"deleted": 2, "deleted_snapshots": 1}
    assert not os.path.exists(snapshot)
    remaining = client.get("/api/logs").json()
    assert [event["name"] for event in remaining] == ["Clip"]


def test_clear_all_logs_removes_persisted_log_file(client, isolated_api_logs):
    res = client.delete("/api/logs")

    assert res.status_code == 200
    assert res.json()["deleted"] == 2
    assert not os.path.exists(os.path.join(isolated_api_logs, "events.json"))
    assert client.get("/api/logs").json() == []


def test_snapshot_missing(client, isolated_api_logs):
    res = client.get("/api/logs/snapshots/does-not-exist.jpg")
    assert res.status_code == 404
