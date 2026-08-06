import pytest
import numpy as np

from tracker_utils import (
    cosine_similarity,
    face_embedding,
    get_color,
    get_face_sys_paths,
    recognize_face,
)


def test_cosine_similarity_identical():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    a = np.zeros(4, dtype=np.float32)
    b = np.ones(4, dtype=np.float32)
    assert cosine_similarity(a, b) == 0.0


def test_face_embedding_prefers_embedding():
    class Face:
        embedding = np.array([1.0, 2.0])
        normed_embedding = np.array([0.1, 0.2])

    emb = face_embedding(Face())
    assert np.allclose(emb, [1.0, 2.0])


def test_face_embedding_falls_back_to_normed():
    class Face:
        normed_embedding = np.array([0.3, 0.4])

    emb = face_embedding(Face())
    assert np.allclose(emb, [0.3, 0.4])


def test_recognize_face_known_match():
    target = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    db = {"Alice": target}
    name, score = recognize_face(target, db, threshold_known=0.35)
    assert name == "Alice"
    assert score == pytest.approx(1.0)


def test_recognize_face_below_threshold():
    query = np.array([1.0, 0.0], dtype=np.float32)
    db = {"Alice": np.array([0.0, 1.0], dtype=np.float32)}
    name, score = recognize_face(query, db, threshold_known=0.35)
    assert name == "Unknown"
    assert score == pytest.approx(0.0)


def test_recognize_face_multi_view():
    query = np.array([1.0, 0.0], dtype=np.float32)
    db = {
        "Bob": np.stack(
            [
                np.array([0.0, 1.0], dtype=np.float32),
                np.array([1.0, 0.0], dtype=np.float32),
            ]
        )
    }
    name, score = recognize_face(query, db)
    assert name == "Bob"
    assert score == pytest.approx(1.0)


def test_recognize_face_empty_db():
    name, score = recognize_face(np.ones(3, dtype=np.float32), {})
    assert name == "Unknown"
    assert score == 0.0


def test_get_color_target_green():
    assert get_color("Alice", "Alice", set()) == (0, 255, 0)


def test_get_color_interacted_cyan():
    assert get_color("Bob", "Alice", {"Bob"}) == (0, 255, 255)


def test_get_color_other_red():
    assert get_color("Eve", "Alice", set()) == (0, 0, 255)


def test_get_face_sys_paths_structure():
    paths = get_face_sys_paths()
    assert "root" in paths
    assert paths["real_db"].endswith("faces_db")
    assert paths["temp_db"].endswith("temp_faces_db")
