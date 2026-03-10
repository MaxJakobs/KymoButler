"""Tests for morphological processing module."""

import numpy as np
import pytest

from kymobutler.morphology import (
    smooth_binary_uni,
    smooth_binary_bi,
    chew_ends,
    chew_all_ends,
    detect_seeds,
    _prune_branches,
    _filter_components,
    process_segmentation_uni,
    process_segmentation_bi,
)


class TestSmoothBinaryUni:
    def test_fills_gaps(self):
        # Create an L-shaped gap that should be filled
        binary = np.zeros((5, 5), dtype=bool)
        binary[1, 2] = True
        binary[2, 3] = True
        binary[3, 2] = True
        result = smooth_binary_uni(binary)
        # Should have filled the center
        assert result[2, 2]

    def test_no_change_on_empty(self):
        binary = np.zeros((5, 5), dtype=bool)
        result = smooth_binary_uni(binary)
        assert not np.any(result)


class TestSmoothBinaryBi:
    def test_output_is_binary(self):
        rng = np.random.default_rng(42)
        binary = rng.random((20, 20)) > 0.7
        result = smooth_binary_bi(binary)
        assert result.dtype == bool


class TestChewEnds:
    def test_removes_horizontal_endpoints(self):
        binary = np.zeros((5, 7), dtype=bool)
        binary[2, 1:6] = True  # horizontal line
        result = chew_ends(binary)
        # Endpoints at (2,1) and (2,5) should be removed
        assert not result[2, 1]
        assert not result[2, 5]
        # Interior points preserved
        assert result[2, 3]

    def test_chew_all_ends_converges(self):
        binary = np.zeros((5, 7), dtype=bool)
        binary[2, 1:6] = True
        result = chew_all_ends(binary)
        # After chewing all ends, nothing should remain for a pure horizontal line
        # (each iteration removes endpoints until nothing left)
        assert not np.any(result) or np.sum(result) <= 1


class TestDetectSeeds:
    def test_finds_top_endpoints(self, simple_skeleton):
        seeds = detect_seeds(simple_skeleton)
        assert len(seeds) > 0
        # Seeds should be sorted by row
        for i in range(len(seeds) - 1):
            assert seeds[i][0] <= seeds[i + 1][0]


class TestPruneBranches:
    def test_removes_short_branches(self, simple_skeleton):
        pruned = _prune_branches(simple_skeleton, iterations=5)
        # The short branch at row 15 should be removed
        assert np.sum(pruned) < np.sum(simple_skeleton)
        # Main vertical line should still exist
        assert np.any(pruned[10:20, 16])


class TestFilterComponents:
    def test_removes_small_components(self):
        binary = np.zeros((50, 50), dtype=bool)
        # Large component
        binary[5:30, 25] = True
        # Small component
        binary[40, 10:12] = True
        result = _filter_components(binary, min_size=5, min_frames=5)
        # Large component should remain
        assert np.any(result[5:30, 25])
        # Small component should be removed
        assert not np.any(result[40, 10:12])
