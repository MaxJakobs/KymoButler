"""Tests for postprocessing module."""

import numpy as np
import pytest

from kymobutler.postprocessing import get_derived_quantities, postprocess, TrackStatistics
from kymobutler.tracking import Track


class TestGetDerivedQuantities:
    def test_basic_track(self):
        t = Track(points=[(0, 0), (1, 1), (2, 2), (3, 3)])
        stats = get_derived_quantities(t)
        assert stats is not None
        assert stats.direction == 1
        assert stats.duration == 4
        assert stats.total_distance == 3.0

    def test_retrograde(self):
        t = Track(points=[(0, 10), (1, 8), (2, 6)])
        stats = get_derived_quantities(t)
        assert stats is not None
        assert stats.direction == -1

    def test_single_point_returns_none(self):
        t = Track(points=[(5, 5)])
        assert get_derived_quantities(t) is None


class TestPostprocess:
    def test_basic_postprocess(self):
        tracks = [
            Track(points=[(0, 0), (1, 2), (2, 4)]),
            Track(points=[(0, 10), (1, 8), (2, 6)]),
        ]
        results = postprocess(tracks, pixel_size_time=0.5, pixel_size_space=0.1)
        assert len(results) == 2
        assert "velocity_um_per_sec" in results[0]
        assert "duration_sec" in results[0]
        assert "distance_um" in results[0]
        assert "direction" in results[0]

    def test_empty_tracks(self):
        results = postprocess([], pixel_size_time=1.0, pixel_size_space=1.0)
        assert results == []
