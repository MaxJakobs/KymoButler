"""Tests for preprocessing module."""

import numpy as np
import pytest

from kymobutler.preprocessing import (
    is_negated,
    normalize_lines,
    resize_to_multiple_of_16,
    load_and_preprocess,
)


class TestIsNegated:
    def test_white_background(self):
        # Mostly white image
        img = np.ones((50, 50), dtype=np.float32) * 0.9
        img[20:30, 20:30] = 0.1
        assert is_negated(img) is True

    def test_black_background(self):
        # Mostly black image
        img = np.ones((50, 50), dtype=np.float32) * 0.1
        img[20:30, 20:30] = 0.9
        assert is_negated(img) is False


class TestNormalizeLines:
    def test_uniform_rows_unchanged(self):
        img = np.ones((10, 10), dtype=np.float32) * 0.5
        result = normalize_lines(img)
        assert result.shape == (10, 10)
        assert result.dtype == np.float32

    def test_zero_rows_preserved(self):
        img = np.zeros((10, 10), dtype=np.float32)
        img[5] = 0.5
        result = normalize_lines(img)
        assert np.all(result[0] == 0.0)  # zero row stays zero


class TestResizeToMultipleOf16:
    def test_already_multiple(self):
        img = np.zeros((64, 128), dtype=np.float32)
        result = resize_to_multiple_of_16(img)
        assert result.shape == (64, 128)

    def test_needs_resize(self):
        img = np.zeros((60, 100), dtype=np.float32)
        result = resize_to_multiple_of_16(img)
        assert result.shape[0] % 16 == 0
        assert result.shape[1] % 16 == 0

    def test_small_image(self):
        img = np.zeros((10, 10), dtype=np.float32)
        result = resize_to_multiple_of_16(img)
        assert result.shape[0] >= 16
        assert result.shape[1] >= 16


class TestLoadAndPreprocess:
    def test_loads_png(self, unitest_path):
        preprocessed, raw, was_negated = load_and_preprocess(unitest_path)
        assert preprocessed.ndim == 2
        assert raw.ndim == 2
        assert preprocessed.dtype == np.float32
        assert 0 <= preprocessed.min()
        assert preprocessed.max() <= 1.0
        assert isinstance(was_negated, bool)

    def test_output_shapes_match(self, bitest_path):
        preprocessed, raw, _ = load_and_preprocess(bitest_path)
        assert preprocessed.shape == raw.shape
