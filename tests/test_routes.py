import pytest
from routes import _parse_classes
from config import CLASSES


class TestParseClasses:
    def test_all_returns_all_classes(self):
        result = _parse_classes("all")
        assert result == CLASSES
        assert len(result) == 9

    def test_empty_string_returns_empty_list(self):
        result = _parse_classes("")
        assert result == []

    def test_single_class(self):
        result = _parse_classes("Car")
        assert result == ["Car"]

    def test_multiple_classes_comma_separated(self):
        result = _parse_classes("Car,Bus,Truck")
        assert result == ["Car", "Bus", "Truck"]

    def test_multiple_classes_with_spaces(self):
        result = _parse_classes("Car, Bus, Truck")
        assert result == ["Car", "Bus", "Truck"]

    def test_whitespace_only_input(self):
        result = _parse_classes("   ")
        assert result == []

    def test_class_case_sensitivity(self):
        result = _parse_classes("car")
        assert result == ["car"]
