"""Tests for tracking module."""

from pathlib import Path

import numpy as np
import pytest

import kymobutler.tracking as tracking
from kymobutler.tracking import (
    Track,
    _clean_up,
    _gather_round_mean,
    _go_back,
    _make_track,
    _resolve_overlaps,
    _Skeleton,
    _split_at_gaps,
    _straddler_mask,
    _straddler_seeds,
    _subset_keep,
    _uni_tracks,
    track_bidirectional,
    track_paths_bidirectional,
    track_unidirectional,
)

DATA = Path(__file__).parent / "data"


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
    """GoBack[x] = Drop[x, i] after While[-i < Length@x && x[[i,1]] - x[[i-1,1]] <= 0, i--]."""

    def test_trims_backwards_run(self):
        # Mathematica: GoBack[{{1,1},{2,1},{3,1},{2,2}}] == {{1,1},{2,1}}
        assert _go_back([(0, 0), (1, 0), (2, 0), (1, 1)]) == [(0, 0), (1, 0)]

    def test_monotone_track_loses_last_point(self):
        # Mathematica: GoBack[{{1,1},{2,1},{3,1},{4,1}}] == {{1,1},{2,1},{3,1}}
        assert _go_back([(0, 0), (1, 0), (2, 0), (3, 0)]) == [(0, 0), (1, 0), (2, 0)]

    def test_all_backwards_drops_everything(self):
        assert _go_back([(3, 0), (2, 0), (2, 1), (1, 1)]) == []

    def test_single_point(self):
        assert _go_back([(4, 4)]) == []


class TestGatherRoundMean:
    """Round/@Mean/@GatherBy[trk, First] on 1-based coordinates."""

    def test_first_appearance_order(self):
        # Mathematica: Round/@Mean/@GatherBy[{{10,1},{11,2},{9,3},{11,4}}, First] == {{10,1},{11,3},{9,3}}
        pts = [(9, 0), (10, 1), (8, 2), (10, 3)]
        assert _gather_round_mean(pts) == [(9, 0), (10, 2), (8, 2)]

    def test_rounds_half_to_even_in_one_based_columns(self):
        # 0-based columns 1, 2 are 1-based 2, 3: Round[5/2] = 2 -> 0-based 1 (0-based rounding gives 2)
        assert _gather_round_mean([(0, 1), (0, 2)]) == [(0, 1)]
        # 0-based 2, 3 are 1-based 3, 4: Round[7/2] = 4 -> 0-based 3
        assert _gather_round_mean([(0, 2), (0, 3)]) == [(0, 3)]
        # exact rational mean: 1-based {1, 2, 2} -> Round[5/3] = 2
        assert _gather_round_mean([(0, 0), (0, 1), (0, 1)]) == [(0, 1)]

    def test_empty(self):
        assert _gather_round_mean([]) == []


class TestUniTracks:
    """UniKymoButlerTrack's track extraction: MorphologicalComponents, PixelValuePositions,
    Round/@Mean/@GatherBy[#, First] and Select[First@Last@# - First@First@# >= minFr]."""

    def test_mean_column_rounded_half_to_even_in_one_based_columns(self):
        skel = np.zeros((6, 8), dtype=bool)
        skel[0, 1:3] = True  # 0-based 1, 2 -> 1-based mean 5/2 -> Round 2 -> 0-based 1
        skel[1, 2:4] = True  # 1-based 3, 4 -> 7/2 -> 4 -> 0-based 3
        skel[2, 3] = True
        skel[3, 3:6] = True  # 1-based 4, 5, 6 -> 5 -> 0-based 4
        assert _uni_tracks(skel, 3) == [[(0, 1), (1, 3), (2, 3), (3, 4)]]

    def test_frame_span_rule(self):
        skel = np.zeros((10, 10), dtype=bool)
        skel[2:6, 1] = True  # spans rows 2..5: last - first = 3
        assert len(_uni_tracks(skel, 3)) == 1
        assert _uni_tracks(skel, 4) == []

    def test_single_row_component_kept_when_min_frames_is_zero(self):
        # Mathematica has no "at least two points" condition
        skel = np.zeros((5, 5), dtype=bool)
        skel[2, 1:3] = True
        assert _uni_tracks(skel, 0) == [[(2, 1)]]

    def test_tracks_in_morphological_components_order(self):
        # arms from (0,0) and (0,6) joined at row 2, dot at (0,3): Mathematica numbers the dot
        # first (the joined arms' root is the right arm's provisional label, see
        # test_morphology), so the dot is track 1 although the arms start further left
        skel = np.array([[1, 0, 0, 1, 0, 0, 1],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 0, 1, 1, 1, 0, 0]], dtype=bool)
        tracks = _uni_tracks(skel, 0)
        assert tracks[0] == [(0, 3)]
        # arms: row 0 mean of 1-based columns 1 and 7 -> 4, row 1 of 2 and 6 -> 4, row 2 -> 4
        assert tracks[1] == [(0, 3), (1, 3), (2, 3)]

    def test_empty(self):
        assert _uni_tracks(np.zeros((4, 4), dtype=bool), 0) == []


class TestSubsetKeep:
    def test_removes_subset(self):
        t1 = [(i, 5) for i in range(20)]
        t2 = [(i, 5) for i in range(5, 10)]
        assert list(_subset_keep([t1, t2], min_size=2)) == [True, False]

    def test_short_track_removed_without_overlap(self):
        # Abs[0 - Length[t2]] <= minSz && Length[t2] < Length[t1]
        t1 = [(i, 5) for i in range(20)]
        t2 = [(i, 50) for i in range(3)]
        assert list(_subset_keep([t1, t2], min_size=3)) == [True, False]
        assert list(_subset_keep([t1, t2], min_size=2)) == [True, True]

    def test_removal_is_simultaneous(self):
        # c is a subset of b, b of a: sequential removal (b first) would keep c; Mathematica drops both
        a = [(i, 0) for i in range(30)]
        b = a[:20]
        c = a[:12]
        assert list(_subset_keep([a, b, c], min_size=2)) == [True, False, False]

    def test_equal_length_kept(self):
        a = [(i, 0) for i in range(10)]
        assert list(_subset_keep([a, list(a)], min_size=3)) == [True, True]

    def test_empty_tracks_removed(self):
        a = [(i, 0) for i in range(10)]
        assert list(_subset_keep([a, []], min_size=0)) == [True, False]


class TestResolveOverlaps:
    def test_winner_is_highest_probability(self):
        a = [(i, 0) for i in range(30)]
        b = [(i, 0) for i in range(10, 40)]
        out = _resolve_overlaps([a, b], [0.5, 0.9])
        assert out[1] == b
        assert out[0] == [(i, 0) for i in range(10)]

    def test_first_wins_ties(self):
        a = [(i, 0) for i in range(30)]
        b = [(i, 0) for i in range(10, 40)]
        out = _resolve_overlaps([a, b], [0.7, 0.7])
        assert out[0] == a
        assert out[1] == [(i, 0) for i in range(30, 40)]

    def test_group_members_lose_winners_points(self):
        # group {0, 1, 2} (1 and 2 each share > 10 points with 0); 1 wins; 2 loses 1's points too
        t0 = [(i, 0) for i in range(40)]
        t1 = [(i, 0) for i in range(0, 20)] + [(i, 5) for i in range(20, 40)]
        t2 = [(i, 0) for i in range(15, 40)]
        out = _resolve_overlaps([t0, t1, t2], [0.1, 0.9, 0.5])
        assert out[1] == t1
        # t2 lost (15..19, 0) to the winner t1 although it shares only 5 points with it; t0 kept
        # (20..39, 0), which still overlaps t2 -> second sweep {0, 2}: t2 wins, t0 is emptied
        assert out[2] == [(i, 0) for i in range(20, 40)]
        assert out[0] == []

    def test_emptied_tracks_are_kept(self):
        a = [(i, 0) for i in range(30)]
        b = [(i, 0) for i in range(5, 20)]
        out = _resolve_overlaps([a, b], [0.9, 0.1])
        assert out == [a, []]


class TestSplitAtGaps:
    def test_splits_at_large_gap(self):
        t = Track(points=[(0, 5), (1, 6), (2, 7), (50, 10), (51, 11)])
        result = _split_at_gaps(t, max_gap=10)
        assert len(result) == 2

    def test_no_split_needed(self):
        t = Track(points=[(0, 5), (1, 6), (2, 7)])
        result = _split_at_gaps(t, max_gap=10)
        assert len(result) == 1


class TestStraddlerSeeds:
    def test_one_seed_per_group(self):
        m = np.zeros((20, 20), bool)
        m[2:12, 3] = True  # seed at (2, 3)
        m[4:12, 6] = True  # seed at (4, 6): same dilation group, lower
        assert _straddler_seeds(m) == [(2, 3)]

    def test_topmost_seeds_of_a_group(self):
        m = np.zeros((20, 20), bool)
        m[2:12, 3] = True
        m[2:12, 6] = True
        assert _straddler_seeds(m) == [(2, 3), (2, 6)]

    def test_seedless_groups_in_mathematica_label_order(self):
        # a U-shaped loop and a rectangular loop between its arms, both without seeds:
        # Mathematica 14.3's CatchStraddlers seeds the rectangle first (its dilation has the
        # smaller MorphologicalComponents label; the U's root is its right arm's label)
        m = np.zeros((40, 45), bool)
        m[5:31, 3] = m[5:31, 37] = m[30, 3:38] = True  # outer outline of the U
        m[5:26, 7] = m[5:26, 33] = m[25, 7:34] = True  # inner outline
        m[5, 3:8] = m[5, 33:38] = True  # tops of the arms
        m[5, 16:25] = m[15, 16:25] = m[5:16, 16] = m[5:16, 24] = True  # rectangle
        assert _straddler_seeds(m) == [(5, 16), (5, 3)]

    def test_straddler_chain_matches_mathematica(self):
        # skeletons and track pixels of bidirectional runs on the public test images (and copies
        # with track pixels removed or added): the tracks leave the skeleton, so paths - tracks
        # has -1 pixels. Mathematica 14.3's pathstmp = SelectComponents[paths - tracks, ...]
        # (BiKymoButlerTrack) and CatchStraddlers' seeds (the package code, MakeTrack stubbed)
        data = np.load(DATA / "mma_straddler_chain.npz")
        paths, tracks, pathstmp = (np.unpackbits(data[k]).astype(bool) for k in ("paths", "tracks", "pathstmp"))
        i = j = 0
        for (h, w), n in zip(data["shapes"], data["nseeds"]):
            p, t, want = (a[i:i + h * w].reshape(h, w) for a in (paths, tracks, pathstmp))
            i += h * w
            assert (t & ~p).any()
            st = _straddler_mask(p, t)
            assert np.array_equal(st, want) and st.sum() > 5
            assert [(r + 1, c + 1) for r, c in _straddler_seeds(st)] == [tuple(q) for q in data["seeds"][j:j + n].tolist()]
            j += n
        assert len(data["shapes"]) == 45 and j == len(data["seeds"])

    def test_key_above_seeds_and_seedless_group(self):
        m = np.zeros((30, 30), bool)
        m[2, 3:9] = True  # horizontal bar: chewAllEnds eats it, no seed
        m[4:14, 5] = True  # seed (4, 5); same dilation group, keyed by the bar's (2, 3)
        m[20, 20:26] = m[25, 20:26] = True  # a closed loop: no seed at all
        m[20:26, 20] = m[20:26, 25] = True
        assert _straddler_seeds(m) == [(2, 3), (20, 20)]


class TestMakeTrack:
    def test_first_step_prefers_largest_row_then_column(self):
        paths = np.zeros((10, 10), bool)
        paths[0, 4] = True
        paths[1, 3] = paths[1, 5] = True  # two candidates in the next row: take column 5
        paths[2:6, 5] = True
        pts, sown = _make_track(_Skeleton(paths), None, (0, 4))
        assert pts[:2] == [(0, 4), (1, 5)]
        assert sown == [1.0]

    def test_lone_seed_sows_nothing(self):
        paths = np.zeros((5, 5), bool)
        paths[2, 2] = True
        assert _make_track(_Skeleton(paths), None, (2, 2)) == ([(2, 2)], None)

    @staticmethod
    def _junction():
        # a vertical line from (0, 5) that forks at (5, 5): two candidates, so GetCand decides
        paths = np.zeros((12, 12), bool)
        paths[0:6, 5] = True
        paths[6, 4] = paths[6, 6] = True
        return paths

    def test_repeating_state_raises_with_the_position(self, monkeypatch):
        # a decision module that keeps answering with points already on the track and no
        # progress in time: MakeTrack's state repeats, so Mathematica's NestWhile never ends
        calls = []

        def stuck(ctx, track):
            calls.append(len(track))
            return [(4, 5), (5, 5)], 0.9

        monkeypatch.setattr(tracking, "get_candidates", stuck)
        with pytest.raises(
            RuntimeError,
            match=r"^KymoButler cannot finish the track that starts at x = 5, y = 0 \(pixels from "
            r"the top-left corner, as ImageJ/Fiji counts them: x is space, y is time\): from "
            r"x = 5, y = 5 on, the decision module keeps sending the track back",
        ):
            _make_track(_Skeleton(self._junction()), object(), (0, 5))
        assert len(calls) == 2  # detected at the first repeat

    def test_whether_the_last_point_repeats_is_part_of_the_state(self, monkeypatch):
        # first a new point (6, 6), then back to (5, 5) and on to (6, 6) again, over and over.
        # After calls 2 and 3 the track ends alike ((5, 5), (6, 6), same points, same
        # backwards_count) but GetCand saw different inputs: at call 2 the track without its
        # last point did not contain (6, 6) yet. Call 4 would repeat call 3 exactly.
        calls = []

        def back_and_forth(ctx, track):
            calls.append(track[-1])
            return ([(6, 6)] if not calls[:-1] else [(5, 5), (6, 6)]), 0.9

        monkeypatch.setattr(tracking, "get_candidates", back_and_forth)
        with pytest.raises(RuntimeError, match="would never finish this kymograph either"):
            _make_track(_Skeleton(self._junction()), object(), (0, 5))
        assert calls == [(5, 5), (6, 6), (6, 6)]

    def test_two_step_cycle_raises(self, monkeypatch):
        # a deterministic decision module that sends the track back and forth between (5, 5)
        # and (4, 5) (one step back in time, one forward, so backwards_count alternates too):
        # the state after the third call equals the one after the first
        calls = []

        def ping_pong(ctx, track):
            calls.append(track[-1])
            return ([(5, 5), (4, 5)] if track[-1] == (5, 5) else [(4, 5), (5, 5)]), 0.9

        monkeypatch.setattr(tracking, "get_candidates", ping_pong)
        with pytest.raises(RuntimeError, match="would never finish"):
            _make_track(_Skeleton(self._junction()), object(), (0, 5))
        assert calls == [(5, 5), (4, 5), (5, 5)]

    def test_revisiting_points_without_repeating_the_state_is_fine(self, monkeypatch):
        # back over old points once, then on to a new one: no cycle
        answers = iter([[(4, 5), (5, 5)], [(6, 6)], []])
        monkeypatch.setattr(tracking, "get_candidates", lambda ctx, track: (next(answers), 0.5))
        pts, sown = _make_track(_Skeleton(self._junction()), object(), (0, 5))
        assert pts[-1] == (6, 6) and sown == [0.5, 0.5, 0.5, 1.0]

    def test_same_end_points_after_new_points_are_not_a_repeat(self, monkeypatch):
        # the second answer ends like the first ((4, 5) then (5, 5)) but adds (6, 6) on the
        # way: the set of track points differs, so the state does not repeat (the track then
        # takes (6, 4), the only free neighbour of (5, 5), and ends at the third answer)
        answers = iter([[(4, 5), (5, 5)], [(6, 6), (4, 5), (5, 5)], []])
        monkeypatch.setattr(tracking, "get_candidates", lambda ctx, track: (next(answers), 0.5))
        pts, sown = _make_track(_Skeleton(self._junction()), object(), (0, 5))
        assert pts[-4:] == [(6, 6), (4, 5), (5, 5), (6, 4)] and sown == [0.5, 0.5, 0.5, 1.0]

    @pytest.mark.parametrize("size, budget, stop", [(12, 1000, "x = 36, y = 27"), (150, 1200, "x = 174, y = 31")])
    def test_decision_budget_is_a_last_resort(self, monkeypatch, size, budget, stop):
        # a decision module that adds one new point per call (row by row through the padding
        # beside the image, so never back in time): the state never repeats, and only the
        # budget, max(4 * (H + W), 1000) calls, stops the track
        calls = []
        cols = [*range(-25, 0), *range(size, size + 25)]
        fresh = iter([(r, c) for r in range(8, size + 25) for c in cols])

        def wander(ctx, track):
            calls.append(1)
            return [next(fresh)], 0.5

        monkeypatch.setattr(tracking, "get_candidates", wander)
        paths = np.zeros((size, size), bool)
        paths[:12, :12] = self._junction()
        with pytest.raises(
            RuntimeError,
            match=rf"^KymoButler stopped tracing the track that starts at x = 5, y = 0 \(pixels "
            rf"from the top-left corner, .*\): it had not ended at {stop} \(just outside the "
            rf"image\) after {budget} decision-module steps \(4 x \(height \+ width\) of this "
            rf"kymograph, at least 1000\)",
        ):
            _make_track(_Skeleton(paths), object(), (0, 5))
        assert len(calls) == budget == tracking._max_vision_calls(size, size)

    def test_long_track_without_decisions_is_not_limited(self):
        # a serpentine through the whole image, one candidate at every step
        paths = np.zeros((40, 40), bool)
        paths[::2, 1:39] = True
        paths[1::4, 39] = paths[3::4, 0] = True
        pts, sown = _make_track(_Skeleton(paths), None, (0, 1))
        assert len(pts) == len(set(pts)) == paths.sum() == 780 and sown == [1.0]


class TestCleanUp:
    def test_decisionless_tracks_score_one(self):
        # Two overlapping tracks: one with a low vision probability, one with none (score 1)
        a = [(i, 0) for i in range(30)]
        b = [(i, 0) for i in range(10, 40)]
        out = _clean_up([a, b], [[0.2, 1.0], [1.0]], (50, 10), min_size=3, min_frames=3)
        assert [t.points for t in out] == [[(i, 0) for i in range(10)], b]

    def test_lone_seed_with_overlaps_raises(self):
        a = [(i, 0) for i in range(30)]
        b = [(i, 0) for i in range(10, 40)]
        with pytest.raises(
            ValueError,
            match=r"with min_size = 1 and min_frames = 1: .* Set min_size \(minimum object size, "
            r"in pixels\) or min_frames \(minimum frames per track\) to 2 or more\.$",
        ):
            _clean_up([a, [(45, 5)], b], [[1.0], None, [1.0]], (50, 10), min_size=1, min_frames=1)

    def test_lone_seed_in_unfiltered_paths_with_overlaps_raises(self):
        # with min_size = 3 only a skeleton that skipped process_segmentation_bi (passed to
        # track_paths_bidirectional directly) can have one-pixel pieces
        a = [(i, 0) for i in range(30)]
        b = [(i, 0) for i in range(10, 40)]
        with pytest.raises(ValueError, match="skeleton with one-pixel pieces, which process_segmentation_bi removes"):
            _clean_up([a, [(45, 5)], b], [[1.0], None, [1.0]], (50, 10), min_size=3, min_frames=3)

    def test_lone_seed_without_overlaps_is_fine(self):
        # Mathematica's Pick fails too, but ptrk is never used: the result is well defined
        a = [(i, 0) for i in range(30)]
        out = _clean_up([a, [(45, 5)]], [[1.0], None], (50, 10), min_size=1, min_frames=1)
        assert [t.points for t in out] == [a]

    def test_split_and_min_frames(self):
        a = [(i, 0) for i in range(5)] + [(i, 0) for i in range(20, 40)]
        out = _clean_up([a], [[1.0]], (50, 10), min_size=3, min_frames=10)
        assert [t.points for t in out] == [[(i, 0) for i in range(20, 40)]]


class TestTrackPaths:
    def test_closed_loop_without_seeds_is_tracked(self):
        # No seed anywhere: Mathematica still tracks it through CatchStraddlers
        paths = np.zeros((40, 40), bool)
        paths[5, 5:20] = paths[30, 5:20] = True
        paths[5:31, 5] = paths[5:31, 19] = True
        kym = np.zeros((40, 40), np.float32)
        tracks = track_paths_bidirectional(paths, kym, 0.5, None, min_size=3, min_frames=3)
        assert len(tracks) >= 1

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            track_paths_bidirectional(np.zeros((5, 5), bool), np.zeros((6, 5), np.float32))


class TestImageSize:
    def test_tiny_kymographs_fail_alike_in_both_pipelines(self):
        msg = "KymoButler needs an image of at least 9 x 9 pixels; this image is 50 x 8"
        with pytest.raises(ValueError, match=msg):
            track_bidirectional(np.zeros((16, 48), np.float32), np.zeros((8, 50), np.float32), False)
        with pytest.raises(ValueError, match=msg):
            track_unidirectional({"ant": np.zeros((16, 48)), "ret": np.zeros((16, 48))}, (8, 50))
