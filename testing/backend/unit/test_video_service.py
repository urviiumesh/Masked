import numpy as np
import pytest

from api.services import video_service


def test_format_time():
    assert video_service._format_time(0) == "00:00:00"
    assert video_service._format_time(65) == "00:01:05"
    assert video_service._format_time(3661) == "01:01:01"


def test_normalize_frame_even_dims():
    frame = np.zeros((101, 201, 3), dtype=np.uint8)
    out = video_service._normalize_frame(frame)
    assert out.shape[0] % 2 == 0
    assert out.shape[1] % 2 == 0
    assert out.dtype == np.uint8
    assert out.flags["C_CONTIGUOUS"]


def test_normalize_frame_gray():
    frame = np.zeros((40, 40), dtype=np.uint8)
    out = video_service._normalize_frame(frame)
    assert len(out.shape) == 3
    assert out.shape[2] == 3


def test_public_job_shape(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"fake")
    job = {
        "id": "abc123",
        "status": "completed",
        "progress": np.float32(100.0),
        "total_frames": 10,
        "current_frame": 10,
        "filename": "clip.mp4",
        "detections": [],
        "error": None,
        "targets": ["Alice"],
        "output_path": None,
        "input_path": str(video),
    }
    public = video_service._public_job(job)
    assert public["id"] == "abc123"
    assert public["can_rescan"] is True
    assert public["progress"] == 100.0
    assert public["output_url"] is None


def test_list_and_get_job_lifecycle(tmp_path, monkeypatch):
    monkeypatch.setattr(video_service, "_jobs", {})
    monkeypatch.setattr(
        video_service.threading,
        "Thread",
        lambda *a, **k: type("T", (), {"start": lambda self: None})(),
    )
    path = str(tmp_path / "v.mp4")
    open(path, "wb").write(b"x")
    job_id = video_service.create_job(path, ["Alice"], "v.mp4")
    jobs = video_service.list_jobs()
    assert any(j["id"] == job_id for j in jobs)
    got = video_service.get_job(job_id)
    assert got is not None
    assert got["targets"] == ["Alice"]
    with video_service._lock:
        video_service._jobs[job_id]["status"] = "completed"
    assert video_service.delete_job(job_id) is True
    assert video_service.get_job(job_id) is None
