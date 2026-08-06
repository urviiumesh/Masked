import numpy as np
import pytest

from api.config import THRESHOLD_HIGH_CONF, THRESHOLD_KNOWN, THRESHOLD_UNKNOWN
from api.services import face_service


def test_thresholds_aligned_with_recognition():
    assert THRESHOLD_KNOWN == 0.35
    assert THRESHOLD_UNKNOWN == 0.35
    assert THRESHOLD_HIGH_CONF == 0.55


def test_person_id_stable():
    assert face_service._person_id("Alice") == face_service._person_id("Alice")
    assert len(face_service._person_id("Alice")) == 5


def test_recognize_embedding_none():
    name, score = face_service.recognize_embedding(None, {"Alice": np.ones(3)})
    assert name == "Unknown"
    assert score == 0.0


def test_recognize_embedding_with_targets():
    emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    db = {
        "Alice": emb,
        "Bob": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    name, score = face_service.recognize_embedding(emb, db, targets={"Bob"})
    assert name == "Unknown"
    name2, score2 = face_service.recognize_embedding(emb, db, targets={"Alice"})
    assert name2 == "Alice"
    assert score2 == pytest.approx(1.0)


def test_recognize_embedding_empty_target_filter():
    emb = np.ones(3, dtype=np.float32)
    name, score = face_service.recognize_embedding(emb, {"Alice": emb}, targets={"Nobody"})
    assert name == "Unknown"
    assert score == 0.0


def test_load_embeddings_for_targets_empty_uses_all(tmp_path, monkeypatch):
    emb_root = tmp_path / "faces_db"
    emb_root.mkdir()
    np.save(emb_root / "Alice.npy", np.ones((1, 4), dtype=np.float32))
    monkeypatch.setattr(face_service, "EMB_DB_ROOT", str(emb_root))
    monkeypatch.setattr(face_service, "TEMP_EMB_DB_ROOT", str(tmp_path / "temp"))
    db = face_service.load_embeddings_for_targets(None)
    assert "Alice" in db


def test_load_embeddings_skips_numeric_gallery_by_default(tmp_path, monkeypatch):
    emb_root = tmp_path / "faces_db"
    emb_root.mkdir()
    np.save(emb_root / "Alice.npy", np.ones((1, 4), dtype=np.float32))
    np.save(emb_root / "12345.npy", np.ones((1, 4), dtype=np.float32))
    monkeypatch.setattr(face_service, "EMB_DB_ROOT", str(emb_root))
    monkeypatch.setattr(face_service, "TEMP_EMB_DB_ROOT", str(tmp_path / "temp"))
    db = face_service.load_all_embeddings(include_numeric_gallery=False)
    assert "Alice" in db
    assert "12345" not in db
    db_all = face_service.load_all_embeddings(include_numeric_gallery=True)
    assert "12345" in db_all


def test_save_upload(tmp_path, monkeypatch):
    monkeypatch.setattr("api.config.UPLOAD_DIR", str(tmp_path))
    path = face_service.save_upload(b"abc", ".jpg")
    assert path.endswith(".jpg")
    assert open(path, "rb").read() == b"abc"
