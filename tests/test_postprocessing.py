"""Tests for kymobutler.postprocessing: getDerivedQuantities and pproc of KymoButlerPProc.wl.

Mathematica's values come from tests/data/reference/pproc_mma.json
(scripts/make_pproc_reference.py): the per-track rows of pprocLocal for the Mathematica tracks of
the public test images and for synthetic tracks (with steps back in time and time gaps), over
many pixel sizes. tests/test_io_utils.py checks the CSV text written from these rows.
"""

import json
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from kymobutler.postprocessing import (
    STAT_KEYS,
    TrackStatistics,
    get_derived_quantities,
    mma_n,
    mma_number,
    mma_round,
    postprocess,
)
from kymobutler.tracking import Track

REF = json.loads((Path(__file__).parent / "data" / "reference" / "pproc_mma.json").read_text())


def _num(s: str):
    """A pixel size as typed in the fixture key: "1" -> 1 (exact), "0.5" -> 0.5."""
    return float(s.replace("*^", "e")) if ("." in s or "*^" in s) else int(s)


def _zero_based(tracks):
    return [[(t - 1, x - 1) for t, x in trk] for trk in tracks]


def _mma_rows(local_csv: str) -> list[list[float]]:
    """The per-track rows of an exported pprocLocal summary (InputForm reals such as 1., 0.3393)."""
    lines = local_csv.splitlines()[2:]
    return [[float(v.replace("*^", "e")) for v in line.split(",")] for line in lines]


class TestGetDerivedQuantities:
    def test_basic_track(self):
        stats = get_derived_quantities(Track(points=[(0, 0), (1, 1), (2, 2), (3, 3)]))
        assert stats == TrackStatistics(
            direction=1, mean_velocity=Fraction(1), total_distance=3, duration=4
        )
        assert stats.start_to_end_velocity == Fraction(3, 4)

    def test_retrograde(self):
        stats = get_derived_quantities(Track(points=[(0, 10), (1, 8), (2, 6)]))
        assert stats.direction == -1
        assert stats.mean_velocity == 2
        assert stats.total_distance == 4

    def test_stationary(self):
        stats = get_derived_quantities([(3, 10), (4, 11), (5, 10)])
        assert stats.direction == 0
        assert stats.total_distance == 2

    def test_time_gap_velocity_is_per_frame(self):
        # |dx/dt| per step: 3/1, 4/2, 0/3 -> mean 5/3
        stats = get_derived_quantities([(0, 0), (1, 3), (3, 7), (6, 7)])
        assert stats.mean_velocity == Fraction(5, 3)
        assert stats.duration == 7

    def test_step_back_in_time_at_the_start(self):
        # Mathematica's first-appearance order: the second point is a frame earlier.
        stats = get_derived_quantities([(9, 4), (8, 6), (10, 7), (11, 7)])
        assert stats.mean_velocity == Fraction(2 + Fraction(1, 2) + 0, 3)  # |dx|/|dt| per step
        assert stats.total_distance == 3
        assert stats.duration == 3  # |t_last - t_first + 1|, not the 4 frames spanned
        assert stats.direction == 1

    def test_step_back_in_time_at_the_end(self):
        stats = get_derived_quantities([(19, 49), (20, 51), (22, 54), (21, 55)])
        assert stats.duration == 3  # last point 21, first 19
        assert stats.direction == 1
        assert stats.total_distance == 6
        assert stats.mean_velocity == Fraction(3, 2)

    def test_numpy_points(self):
        pts = np.array([[0, 5], [1, 3], [2, 3]], dtype=np.int64)
        assert get_derived_quantities(pts) == get_derived_quantities([(0, 5), (1, 3), (2, 3)])

    def test_single_point_returns_none(self):
        assert get_derived_quantities(Track(points=[(5, 5)])) is None
        assert get_derived_quantities([]) is None

    def test_same_frame_twice_raises(self):
        with pytest.raises(ValueError, match="same frame"):
            get_derived_quantities([(0, 0), (1, 1), (1, 2)])

    def test_fractional_points_raise(self):
        with pytest.raises(ValueError, match="whole pixel"):
            get_derived_quantities([(0, 0), (1, 1.5)])


class TestPostprocessMatchesMathematica:
    @pytest.mark.parametrize("key", sorted(REF["csv"]))
    def test_rows(self, key):
        name, ts, xs = key.split("|")
        rows = postprocess(_zero_based(REF["sets"][name]), _num(ts), _num(xs))
        want = _mma_rows(REF["csv"][key]["local"])
        got = [[row[k] for k in STAT_KEYS] for row in rows]
        assert got == want  # bit for bit, including Mathematica's binary digits
        assert all(type(v) is float for r in got for v in r)

    def test_backward_steps_are_in_the_reference(self):
        # the synthetic set exercises steps back in time at the start, middle and end
        syn = REF["sets"]["synthetic"]
        assert sum(b[0] < a[0] for t in syn for a, b in zip(t, t[1:])) >= 10
        assert any(t[-1][0] < max(p[0] for p in t) for t in syn)
        assert any(t[0][0] > min(p[0] for p in t) for t in syn)

    def test_mathematica_digits(self):
        # bi_bitest_t0.2_s10_f10, track 3, pixel sizes 1 and 1: Round[.., 0.0001] keeps binary
        # noise
        rows = postprocess(_zero_based(REF["sets"]["bi_bitest_t0.2_s10_f10"]), 1, 1)
        assert rows[2]["velocity_um_per_sec"] == 0.36510000000000004
        assert rows[2]["direction"] == -1.0

    def test_integral_float_pixel_sizes_are_exact_integers(self):
        tracks = _zero_based(REF["sets"]["synthetic"])
        assert postprocess(tracks, 1.0, 3.0) == postprocess(tracks, 1, 3)
        assert postprocess(tracks, np.float32(0.5), np.int64(2)) == postprocess(tracks, 0.5, 2)


class TestPostprocess:
    def test_keys_in_summary_order(self):
        rows = postprocess([Track(points=[(0, 0), (1, 2), (2, 4)])], 0.5, 0.1)
        assert tuple(rows[0]) == STAT_KEYS

    def test_values(self):
        tracks = [Track(points=[(0, 0), (1, 2), (2, 4)]), Track(points=[(0, 10), (1, 8), (2, 6)])]
        rows = postprocess(tracks, 0.5, 0.1)
        assert rows[0] == {
            "direction": 1.0,
            "velocity_um_per_sec": 0.4,  # 0.1/0.5 * 2 px/frame
            "duration_sec": 1.5,  # 3 frames
            "distance_um": 0.4,
            "start2end_velocity_um_per_sec": mma_round(0.1 * ((4 * (1 / 3)) * (1 / 0.5))),
        }
        assert rows[1]["direction"] == -1.0

    def test_empty_tracks(self):
        assert postprocess([], pixel_size_time=1.0, pixel_size_space=1.0) == []

    def test_short_track_raises_with_its_number(self):
        with pytest.raises(ValueError, match="track #2: .*fewer than two points"):
            postprocess([[(0, 0), (1, 1)], [(4, 4)]])

    def test_zero_duration_raises(self):
        # T = t_last - t_first + 1 = 0: Mathematica divides by zero (ComplexInfinity)
        with pytest.raises(ValueError, match="track #1: .*duration"):
            postprocess([[(5, 1), (4, 2)]])

    @pytest.mark.parametrize("bad", [0, -1, float("nan"), float("inf")])
    def test_bad_pixel_size_raises(self, bad):
        with pytest.raises(ValueError, match="positive"):
            postprocess([[(0, 0), (1, 1)]], bad, 1)
        with pytest.raises(ValueError, match="positive"):
            postprocess([[(0, 0), (1, 1)]], 1, bad)

    @pytest.mark.parametrize("bad", [True, "1", None])
    def test_non_number_pixel_size_raises(self, bad):
        with pytest.raises(TypeError):
            postprocess([[(0, 0), (1, 1)]], bad, 1)


class TestNumbers:
    def test_mma_round_is_round_times_step_ties_to_even(self):
        assert mma_round(0.3651) == 3651 * 0.0001 == 0.36510000000000004
        assert mma_round(0.00025) == 0.0002  # Round[2.5] = 2
        assert mma_round(0.00035) == 0.0004
        assert mma_round(281.0) == 281.0

    @pytest.mark.parametrize(
        "x,want",  # Round[x, 0.0001] as pproc evaluates it (Mathematica 14.3)
        [(0.00155, 0.0015), (0.00195, 0.0019), (0.00275, 0.0027), (0.00295, 0.0029000000000000002),
         (0.00305, 0.003), (0.00435, 0.0043)],
    )
    def test_mma_round_near_ties(self, x, want):
        # x / 0.0001 in machine arithmetic (15.499999999999998 for 0.00155), not x * 10000.
        assert mma_round(x) == want

    @pytest.mark.parametrize(
        "p,q,want",  # N[p/q] (Mathematica 14.3): N[p] / N[q], not the correctly rounded quotient
        [(1395682427949473838310353, 2143691540712320948, 651064.951017023),
         (95494603668627367801135, 1798754204912731229, 53089.301143988385),
         (893341278484494181837373, 2267139777616210584, 394038.9063367764)],
    )
    def test_mma_n_of_big_rationals(self, p, q, want):
        assert mma_n(Fraction(p, q)) == want
        assert float(Fraction(p, q)) != want

    def test_mma_n_of_small_rationals_is_correctly_rounded(self):
        assert mma_n(Fraction(5, 3)) == 5 / 3 and mma_n(7) == 7.0 and type(mma_n(7)) is float

    def test_mma_number(self):
        assert mma_number(1.0) == 1 and type(mma_number(1.0)) is int
        assert type(mma_number(np.int32(3))) is int
        assert mma_number(0.5) == 0.5 and type(mma_number(np.float64(0.5))) is float
