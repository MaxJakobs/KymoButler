"""Tests for tracking module."""

import numpy as np
import pytest

from kymobutler.tracking import (
    Track,
    _go_back,
    _average_duplicates,
    _remove_subset_tracks,
    _split_at_gaps,
)


class TestTrack:
    def test_duration(self):
        t = Track(points=[(0, 5), (5, 10), (10, 15)])
        assert t.duration == 10

    def test_direction_positive(self):
        t = Track(points=[(0, 5), (5, 10)])
        assert t.direction == 1

    def test_direction_negative(self):
        t = Track(points=[(0, 10), (5, 5)])
        assert t.direction == -1


class TestGoBack:
    def test_trims_backwards(self):
        points = [(0, 5), (1, 6), (2, 7), (1, 8)]  # last point goes back in time
        result = _go_back(points)
        assert len(result) < len(points)

    def test_no_trim_needed(self):
        points = [(0, 5), (1, 6), (2, 7)]
        result = _go_back(points)
        assert len(result) == len(points)


class TestAverageDuplicates:
    def test_averages_same_timepoint(self):
        points = [(1, 5), (1, 7), (2, 10)]
        result = _average_duplicates(points)
        assert len(result) == 2
        assert result[0] == (1, 6)  # mean of 5, 7
        assert result[1] == (2, 10)

    def test_empty(self):
        assert _average_duplicates([]) == []


class TestRemoveSubsetTracks:
    def test_removes_subset(self):
        t1 = Track(points=[(i, 5) for i in range(20)])
        t2 = Track(points=[(i, 5) for i in range(5, 10)])  # subset of t1
        result = _remove_subset_tracks([t1, t2], min_size=2)
        assert len(result) == 1
        assert len(result[0].points) == 20


class TestSplitAtGaps:
    def test_splits_at_large_gap(self):
        t = Track(points=[(0, 5), (1, 6), (2, 7), (50, 10), (51, 11)])
        result = _split_at_gaps(t, max_gap=10)
        assert len(result) == 2

    def test_no_split_needed(self):
        t = Track(points=[(0, 5), (1, 6), (2, 7)])
        result = _split_at_gaps(t, max_gap=10)
        assert len(result) == 1
