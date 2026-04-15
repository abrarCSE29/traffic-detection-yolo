import numpy as np
import pytest
import cv2


@pytest.fixture
def sample_frame():
    """Create a dummy 640x480 RGB frame for testing."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return frame


@pytest.fixture
def sample_boxes():
    """Sample bounding boxes with labels and colors."""
    return [
        {"box": [100, 100, 200, 200], "label": "Car #1", "color": (0, 255, 0)},
        {"box": [300, 150, 400, 250], "label": "Bus #2", "color": (255, 0, 0)},
    ]
