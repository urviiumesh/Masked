def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["service"] == "dhrishti"
    assert "ort_provider" in body
    assert "available_providers" in body
    assert isinstance(body["gpu_enabled"], bool)
