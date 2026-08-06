import os

from api.config import EMB_DB_ROOT


def test_list_presets(client):
    res = client.get("/api/stream/presets")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body, list)
    assert len(body) >= 1
    preset = body[0]
    assert "id" in preset
    assert "name" in preset
    assert "host" in preset
    assert "password" not in preset
    assert "username" not in preset


def test_stream_status_disconnected(client):
    client.post("/api/stream/disconnect")
    res = client.get("/api/stream/status")
    assert res.status_code == 200
    body = res.json()
    assert body["connected"] is False
    assert "active_targets" in body
    assert "fps" in body


def test_connect_requires_source_or_preset(client):
    res = client.post("/api/stream/connect", json={"location": "Test"})
    assert res.status_code == 400


def test_set_targets(client):
    known = [
        name.replace(".npy", "")
        for name in os.listdir(EMB_DB_ROOT)
        if name.endswith(".npy") and not name.replace(".npy", "").isdigit()
    ]
    assert known, "Expected at least one named embedding in faces_db"
    chosen = known[:2]
    res = client.put("/api/stream/targets", json=chosen)
    assert res.status_code == 200
    body = res.json()
    assert set(body["active_targets"]) == set(chosen)


def test_disconnect(client):
    res = client.post("/api/stream/disconnect")
    assert res.status_code == 200
    assert res.json()["connected"] is False


def test_snapshot_without_stream(client):
    client.post("/api/stream/disconnect")
    res = client.get("/api/stream/snapshot")
    assert res.status_code == 404
