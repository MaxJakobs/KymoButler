"""Tests for the vision-module wrapper (GetTile / GetCandFromPmap / SortCoords / GetCand)."""

import numpy as np
import torch

from kymobutler.graph_utils import find_shortest_path_on_skeleton
from kymobutler.vision_module import (
    VisionContext,
    get_candidates,
    get_candidates_from_pmap,
    sort_coords,
)


class FakeNet(torch.nn.Module):
    """Returns a fixed foreground probability map and records its inputs."""

    def __init__(self, pmap: np.ndarray):
        super().__init__()
        self.pmap = pmap.astype(np.float32)
        self.calls = []

    def forward(self, tile, trk, struct):
        self.calls.append((tile.numpy()[0, 0].copy(), trk.numpy()[0, 0].copy(), struct.numpy()[0, 0].copy()))
        out = torch.zeros(1, 2, 48, 48)
        out[0, 1] = torch.from_numpy(self.pmap)
        return out


class TestSortCoords:
    def test_equal_length_chains_take_canonically_larger(self):
        # outL = {(5,5),(4,6)}, outR = {(5,5),(6,6)}: Last@SortBy[{outR,outL},Length] -> outR
        assert sort_coords([(5, 5), (4, 6), (6, 6)]) == [(5, 5), (6, 6)]

    def test_column_ties_break_by_row(self):
        # From (5,5) the neighbours (4,6), (6,6) tie on the column: First@SortBy -> (4,6),
        # Last@SortBy -> (6,6); the right chain continues from (6,6) and is longer.
        pts = [(5, 5), (4, 6), (6, 6), (7, 7), (8, 8)]
        assert sort_coords(pts) == [(5, 5), (6, 6), (7, 7), (8, 8)]

    def test_longer_chain_wins(self):
        pts = [(5, 5), (5, 4), (5, 6), (5, 7), (5, 8)]
        assert sort_coords(pts) == [(5, 5), (5, 6), (5, 7), (5, 8)]


class TestShortestPath:
    def test_unweighted_bfs_in_label_order(self):
        # From (1,0) to (1,2): the diagonal route via (0,1) (label 3) and the straight route via
        # (1,1) have the same number of hops; Mathematica's unweighted FindShortestPath returns the
        # one through the lowest label, not the geometrically shorter one.
        b = np.zeros((3, 3), bool)
        b[0, 1] = b[1, 1] = b[2, 1] = True
        assert find_shortest_path_on_skeleton(b, (1, 0), (1, 2)) == [(1, 0), (0, 1), (1, 2)]

    def test_unconnected(self):
        b = np.zeros((5, 5), bool)
        assert find_shortest_path_on_skeleton(b, (0, 0), (4, 4)) == []


class TestCandidatesFromPmap:
    def test_largest_component_and_probability(self):
        pmap = np.zeros((48, 48), np.float32)
        pmap[10:20, 5] = 0.8
        pmap[30:33, 30] = 0.9
        cands, prob = get_candidates_from_pmap(pmap, 0.5)
        assert cands == [(r, 5) for r in range(10, 20)]
        assert abs(prob - 0.8) < 1e-6

    def test_nothing_above_threshold(self):
        assert get_candidates_from_pmap(np.full((48, 48), 0.5, np.float32), 0.5) == ([], None)


def _context(paths, pmap):
    net = FakeNet(pmap)
    kym = np.random.default_rng(0).random(paths.shape).astype(np.float32)
    return VisionContext(kym, paths, net, threshold=0.5), net


class TestGetCandidates:
    # The track ends ..., S=(40,40), L=(41,40). GetCand centres the tile on S, so tile (i, j) is
    # image (i + 16, j + 16) and S is tile (24, 24).

    def test_tile_centred_on_second_to_last_point_and_track_candidates_kept(self):
        paths = np.zeros((100, 100), bool)
        paths[20:90, 40] = True
        pmap = np.zeros((48, 48), np.float32)
        pmap[24:41, 24] = 0.9  # includes S's own tile position (24, 24)
        ctx, net = _context(paths, pmap)
        new, prob = get_candidates(ctx, [(39, 40), (40, 40), (41, 40)])
        _, trk, struct = net.calls[0]
        assert {tuple(p) for p in np.argwhere(trk)} == {(23, 24), (24, 24)}  # track without L
        assert {tuple(p) for p in np.argwhere(struct)} == {(r, 24) for r in range(4, 48)}
        # Complement[cands, trktmp] keeps S (tile coordinates never equal padded ones) and the
        # candidates are ordered from lastTrk = S
        assert new == [(r, 40) for r in range(40, 57)]
        assert abs(prob - 0.9) < 1e-6

    def test_falls_back_to_full_chain_when_bridge_goes_backwards(self):
        paths = np.zeros((100, 100), bool)
        paths[16:41, 40] = True  # tile column 24, rows 0..24
        paths[16, 40:47] = True  # tile row 0
        paths[16:41, 46] = True  # tile column 30, rows 0..24
        pmap = np.zeros((48, 48), np.float32)
        pmap[20:48, 30] = 0.9
        ctx, _ = _context(paths, pmap)
        new, prob = get_candidates(ctx, [(39, 40), (40, 40), (41, 40)])
        # the bridge S -> (24,30) climbs the skeleton, so the first 24 points of path+chain average
        # far above S; Mathematica then returns the SortCoords chain itself
        assert new == [(r, 46) for r in range(40, 64)]
        assert prob is not None

    def test_probability_sown_even_when_candidates_rejected(self):
        paths = np.zeros((100, 100), bool)
        paths[20:90, 40] = True
        pmap = np.zeros((48, 48), np.float32)
        pmap[30:32, 24] = 0.7  # two candidates only: rejected
        ctx, _ = _context(paths, pmap)
        new, prob = get_candidates(ctx, [(39, 40), (40, 40), (41, 40)])
        assert new == []
        assert abs(prob - 0.7) < 1e-6

    def test_no_skeleton_nearby_skips_the_net(self):
        paths = np.zeros((300, 300), bool)
        paths[250:260, 250] = True
        ctx, net = _context(paths, np.ones((48, 48), np.float32))
        assert get_candidates(ctx, [(39, 40), (40, 40), (41, 40)]) == ([], None)
        assert net.calls == []

    def test_constant_tile_is_zero(self):
        paths = np.zeros((100, 100), bool)
        paths[20:90, 40] = True
        net = FakeNet(np.zeros((48, 48), np.float32))
        ctx = VisionContext(np.full((100, 100), 0.1, np.float32), paths, net, threshold=0.5)
        get_candidates(ctx, [(39, 40), (40, 40), (41, 40)])
        assert not net.calls[0][0].any()  # ImageAdjust of a constant image is 0
