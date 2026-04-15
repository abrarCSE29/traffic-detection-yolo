import pytest
from config import CLASSES, CLASS_COLORS, CONF_THRESHOLD, IOU_THRESHOLD, TARGET_FPS


class TestConfig:
    def test_classes_count(self):
        assert len(CLASSES) == 9

    def test_class_colors_count(self):
        assert len(CLASS_COLORS) == 9

    def test_all_classes_have_colors(self):
        for cls in CLASSES:
            assert cls in CLASS_COLORS, f"Missing color for {cls}"

    def test_conf_threshold_positive(self):
        assert 0 < CONF_THRESHOLD < 1

    def test_iou_threshold_positive(self):
        assert 0 < IOU_THRESHOLD < 1

    def test_target_fps_positive(self):
        assert TARGET_FPS > 0

    def test_classes_are_valid(self):
        expected = [
            "Bike",
            "Bus",
            "Car",
            "Cng",
            "People",
            "Rickshaw",
            "Truck",
            "Mini-Truck",
            "Cycle",
        ]
        assert CLASSES == expected
