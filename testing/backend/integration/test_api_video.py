from io import BytesIO


def test_list_video_jobs(client):
    res = client.get("/api/video/jobs")
    assert res.status_code == 200
    assert isinstance(res.json(), list)


def test_get_video_job_not_found(client):
    res = client.get("/api/video/jobs/missingjobid")
    assert res.status_code == 404


def test_upload_requires_targets(client):
    files = {"video": ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")}
    data = {"targets": ""}
    res = client.post("/api/video/upload", files=files, data=data)
    assert res.status_code == 400


def test_upload_and_delete_job(client, monkeypatch):
    from api.services import video_service

    created = {}

    def fake_create(path, targets, filename=""):
        job_id = "testjob001"
        with video_service._lock:
            video_service._jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "progress": 0.0,
                "total_frames": 0,
                "current_frame": 0,
                "output_path": None,
                "input_path": path,
                "detections": [],
                "error": None,
                "targets": targets or [],
                "filename": filename,
            }
        created["id"] = job_id
        return job_id

    monkeypatch.setattr(video_service, "create_job", fake_create)

    files = {"video": ("clip.mp4", BytesIO(b"fake-video-bytes"), "video/mp4")}
    data = {"targets": "Alice,Bob"}
    res = client.post("/api/video/upload", files=files, data=data)
    assert res.status_code == 200
    body = res.json()
    assert body["job_id"] == "testjob001"
    assert body["targets"] == ["Alice", "Bob"]

    got = client.get("/api/video/jobs/testjob001")
    assert got.status_code == 200
    assert got.json()["filename"] == "clip.mp4"

    with video_service._lock:
        video_service._jobs["testjob001"]["status"] = "completed"

    deleted = client.delete("/api/video/jobs/testjob001")
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] == "testjob001"


def test_rescan_requires_targets(client):
    res = client.post("/api/video/jobs/missing/rescan", json={"targets": []})
    assert res.status_code == 400
