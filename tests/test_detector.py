import pytest
import cv2
import numpy as np
from detector import visualize_and_encode


class TestVisualizeAndEncode:
    def test_empty_boxes_returns_valid_jpeg(self, sample_frame):
        result = visualize_and_encode(sample_frame, [])
        assert isinstance(result, bytes)
        assert result.startswith(b"\xff\xd8")  # JPEG magic bytes

    def test_single_box_draws_rectangle(self, sample_frame):
        boxes = [
            {"box": [100, 100, 200, 200], "label": "Car #1", "color": (0, 255, 0)},
        ]
        result = visualize_and_encode(sample_frame, boxes, jpeg_quality=90)
        assert isinstance(result, bytes)
        assert result.startswith(b"\xff\xd8")

    def test_multiple_boxes(self, sample_frame, sample_boxes):
        result = visualize_and_encode(sample_frame, sample_boxes)
        assert isinstance(result, bytes)
        assert result.startswith(b"\xff\xd8")

    def test_jpeg_quality_affects_size(self, sample_frame, sample_boxes):
        low_quality = visualize_and_encode(sample_frame, sample_boxes, jpeg_quality=10)
        high_quality = visualize_and_encode(
            sample_frame, sample_boxes, jpeg_quality=100
        )
        assert len(low_quality) < len(high_quality)

    def test_frame_modified_in_place(self, sample_frame):
        original = sample_frame.copy()
        boxes = [{"box": [100, 100, 200, 200], "label": "Test", "color": (0, 255, 0)}]
        visualize_and_encode(sample_frame, boxes)
        assert not np.array_equal(sample_frame, original)
