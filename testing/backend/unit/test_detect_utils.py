import numpy as np
import pytest

from api.services.detect_utils import (
    _iou,
    _nms_faces,
    _round_down_even,
    prepare_detect_frame,
)


class _FakeFace:
    def __init__(self, bbox, score=0.9):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.det_score = score
        self.kps = None


def test_round_down_even():
    assert _round_down_even(5) == 4
    assert _round_down_even(4) == 4
    assert _round_down_even(1) == 2


def test_prepare_detect_frame_no_downscale():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    out, scale = prepare_detect_frame(frame, max_width=960)
    assert out.shape[1] <= 200
    assert scale == pytest.approx(1.0) or out.shape[1] % 2 == 0


def test_prepare_detect_frame_downscales_wide():
    frame = np.zeros((720, 1920, 3), dtype=np.uint8)
    out, scale = prepare_detect_frame(frame, max_width=960)
    assert out.shape[1] <= 960
    assert out.shape[1] % 2 == 0
    assert out.shape[0] % 2 == 0
    assert scale < 1.0


def test_iou_identical():
    a = np.array([0, 0, 10, 10], dtype=np.float32)
    assert _iou(a, a) == pytest.approx(1.0)


def test_iou_no_overlap():
    a = np.array([0, 0, 10, 10], dtype=np.float32)
    b = np.array([20, 20, 30, 30], dtype=np.float32)
    assert _iou(a, b) == 0.0


def test_nms_faces_keeps_higher_score():
    faces = [
        _FakeFace([0, 0, 10, 10], score=0.5),
        _FakeFace([1, 1, 11, 11], score=0.9),
        _FakeFace([50, 50, 60, 60], score=0.8),
    ]
    kept = _nms_faces(faces, iou_thresh=0.3)
    assert len(kept) == 2
    assert kept[0].det_score == 0.9
