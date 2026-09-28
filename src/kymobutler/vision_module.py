"""Vision (decision) module inference and candidate extraction.

Corresponds to GetTile, GetCandFromPmap, SortCoords and GetCand in KymoButler.wl lines 128-243.

Coordinates are 0-based (row, col) tuples. Mathematica works 1-based; every step here is
translation invariant (differences, distances, canonical lexicographic tie-breaks), so the
0-based computation reproduces it exactly.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import label
from scipy.spatial import KDTree
from skimage.morphology import thin

from kymobutler.config import MAX_CANDIDATES, VISION_MODULE_TILE_SIZE
from kymobutler.graph_utils import find_shortest_path_on_skeleton

_TILE = VISION_MODULE_TILE_SIZE  # 48
_PAD = round(1 + _TILE / 2)  # Round[1 + dim/2] = 25
_NEAR_RADIUS = _TILE * 1.5  # Nearest[allyx, Last@trk, {All, dim*1.5}]
_NEIGHBOURS_8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
_STRUCT_8 = np.ones((3, 3), dtype=int)


class VisionContext:
    """Per-kymograph state for GetCand: the padded kymograph and skeleton, built once."""

    def __init__(
        self,
        kym: np.ndarray,
        paths: np.ndarray,
        vision_net: torch.nn.Module,
        threshold: float,
        device: str = "cpu",
    ):
        self.h, self.w = kym.shape
        # padkym = ArrayPad[ImageData@kym, Round[1 + dim/2], .1]
        self.padkym = np.pad(kym.astype(np.float32), _PAD, mode="constant", constant_values=0.1)
        self.padpaths = np.pad(np.asarray(paths, dtype=bool), _PAD)
        self.net = vision_net
        self.net.eval()
        self.threshold = threshold
        self.device = device
        self._kdtree: KDTree | None = None
        self.n_calls = 0  # vision-module evaluations

    def any_path_pixel_near(self, point: tuple[int, int]) -> bool:
        """Is any skeleton pixel within 72 px (inclusive) of ``point``?"""
        if self._kdtree is None:
            coords = np.argwhere(self.padpaths[_PAD:-_PAD, _PAD:-_PAD]).astype(np.float64)
            if len(coords) == 0:
                return False
            self._kdtree = KDTree(coords)
        return self._kdtree.query_ball_point(point, r=_NEAR_RADIUS, return_length=True) > 0


def get_candidates_from_pmap(
    pmap: np.ndarray, threshold: float
) -> tuple[list[tuple[int, int]], float | None]:
    """Candidates from the vision-module probability map (GetCandFromPmap, KymoButler.wl 164-171).

    Largest 8-connected component of ``pmap > threshold`` (the first one on ties), thinned.

    Returns:
        (candidates, decision_prob). Candidates are (row, col) in lexicographic order.
        decision_prob is the mean of ``pmap`` over that component; Mathematica Sows it to
        "DecisionProb" whenever a component exists, whatever GetCand later does with the
        candidates. It is None (nothing Sown) when no pixel exceeds the threshold.
    """
    labeled, num = label(pmap > threshold, structure=_STRUCT_8)
    if num == 0:
        return [], None
    counts = np.bincount(labeled.ravel())[1:]
    mask = labeled == (int(np.argmax(counts)) + 1)  # MaxIndx: first maximum
    decision_prob = float(pmap[mask].astype(np.float64).sum() / mask.sum())
    coords = [(int(r), int(c)) for r, c in np.argwhere(thin(mask))]
    return coords, decision_prob


def sort_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Order candidates into a chain (SortCoords, KymoButler.wl 128-146).

    From the first point, repeatedly step to a remaining point within distance 1.5 (an
    8-neighbour): the "left" chain takes ``First@SortBy[foo, Last]`` (minimum column, ties by
    minimum row: SortBy breaks ties in canonical order), the "right" chain
    ``Last@SortBy[foo, Last]`` (maximum column, then maximum row). Returns
    ``Last@SortBy[{outR, outL}, Length]``: the longer chain, the canonically larger one on
    equal length.
    """
    if len(coords) <= 1:
        return list(coords)
    first = coords[0]

    def grow(pick_max: bool) -> list[tuple[int, int]]:
        remaining = set(coords[1:])
        out = [first]
        while remaining:
            r, c = out[-1]
            nbrs = [(r + dr, c + dc) for dr, dc in _NEIGHBOURS_8 if (r + dr, c + dc) in remaining]
            if not nbrs:
                break
            key = lambda p: (p[1], p[0])  # noqa: E731
            pick = max(nbrs, key=key) if pick_max else min(nbrs, key=key)
            out.append(pick)
            remaining.discard(pick)
        return out

    out_l = grow(pick_max=False)
    out_r = grow(pick_max=True)
    return max([out_r, out_l], key=lambda o: (len(o), o))


def _rows_mean_ok(points: list[tuple[int, int]], last_row: int) -> bool:
    """Mean[points[[;;,1]]] - First@lastTrk >= -1, in exact integer arithmetic."""
    n = len(points)
    return sum(p[0] for p in points) - n * last_row >= -n


def get_candidates(
    ctx: VisionContext,
    track: list[tuple[int, int]],
) -> tuple[list[tuple[int, int]], float | None]:
    """Use the vision module to find the next coordinates of a track (GetCand, KymoButler.wl 183-243).

    Args:
        ctx: Per-kymograph context (padded kymograph/skeleton, net, threshold).
        track: Current track (at least 3 points), 0-based (row, col).

    Returns:
        (new_coordinates, decision_prob): the coordinates to append (0-based, possibly outside
        the image) and the probability Mathematica Sows for this call (None if it Sows
        nothing).
    """
    dim, pad = _TILE, _PAD
    last = track[-1]
    # allyxtmp = Nearest[allyx + pad, Last@trktmp, {All, 72}]; If[Length@allyxtmp > 0, ..., {}]
    # trktmp = Drop[trktmp, -1]: the tile, lastTrk and the Complement use the track without its
    # last point, so the tile is centred on the second-to-last point.
    s_r, s_c = track[-2]

    # GetTile window (Round[Last@trk - dim/2] .. Round[Last@trk + dim/2 - 1], shifted inside).
    hp, wp = ctx.padkym.shape
    r0 = s_r + pad - dim // 2
    c0 = s_c + pad - dim // 2
    if r0 < 0:
        r0 = 0
    elif r0 + dim > hp:
        r0 = hp - dim
    if c0 < 0:
        c0 = 0
    elif c0 + dim > wp:
        c0 = wp - dim

    # fullbin: skeleton pixels within 72 px of the (true) last point, cropped to the window.
    fullbin = ctx.padpaths[r0:r0 + dim, c0:c0 + dim].copy()
    lr, lc = last[0] + pad - r0, last[1] + pad - c0  # last point in tile coordinates
    farthest_sq = max(lr ** 2, (dim - 1 - lr) ** 2) + max(lc ** 2, (dim - 1 - lc) ** 2)
    if farthest_sq > _NEAR_RADIUS ** 2:  # some tile pixel is farther than 72 px from it
        rr, cc = np.ogrid[:dim, :dim]
        fullbin &= (rr - lr) ** 2 + (cc - lc) ** 2 <= _NEAR_RADIUS ** 2
    if not fullbin.any() and not ctx.any_path_pixel_near(last):
        return [], None

    # tile: ImageAdjust@ImageTake[padkym, win] (ImageAdjust maps a constant image to 0)
    tile = ctx.padkym[r0:r0 + dim, c0:c0 + dim].astype(np.float64)
    tmin, tmax = tile.min(), tile.max()
    tile = (tile - tmin) / (tmax - tmin) if tmax > tmin else np.zeros_like(tile)
    tile = tile.astype(np.float32)

    # bin: the track without its last point, cropped to the window.
    trk = np.asarray(track[:-1], dtype=np.int64) + pad - np.array([r0, c0])
    inside = (trk[:, 0] >= 0) & (trk[:, 0] < dim) & (trk[:, 1] >= 0) & (trk[:, 1] < dim)
    trkbin = np.zeros((dim, dim), dtype=np.float32)
    trkbin[trk[inside, 0], trk[inside, 1]] = 1.0

    def as_tensor(a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))[None, None].to(ctx.device)

    with torch.no_grad():
        pmap = ctx.net(as_tensor(tile), as_tensor(trkbin), as_tensor(fullbin))[0, 1].cpu().numpy()
    ctx.n_calls += 1

    cands, decision_prob = get_candidates_from_pmap(pmap, ctx.threshold)
    if not cands:
        return [], decision_prob

    # cands = Complement[cands, trktmp]: Mathematica compares the tile coordinates of the
    # candidates with the *padded full-image* coordinates of the track, so this only sorts,
    # de-duplicates and drops the (rare) candidates whose tile coordinates coincide with a
    # padded track coordinate. Candidates on the existing track are kept.
    padded_track = np.asarray(track[:-1], dtype=np.int64) + pad
    near_origin = padded_track[(padded_track[:, 0] < dim) & (padded_track[:, 1] < dim)]
    if len(near_origin):
        drop = {(int(r), int(c)) for r, c in near_origin}
        cands = [p for p in cands if p not in drop]

    # lastTrk = Last@trktmp in tile coordinates (the second-to-last track point).
    t_r, t_c = s_r + pad - r0, s_c + pad - c0
    cands.sort(key=lambda p: ((p[0] - t_r) ** 2 + (p[1] - t_c) ** 2, p))
    if not (len(cands) > 2 and _rows_mean_ok(cands, t_r)):
        return [], decision_prob

    cands = sort_coords(cands)
    select = cands
    if (cands[0][0] - t_r) ** 2 + (cands[0][1] - t_c) ** 2 > 1.5 ** 2:
        # FindShortPathImage over the skeleton pixels of the tile, lastTrk -> First@cands
        # (the path includes both ends).
        select = find_shortest_path_on_skeleton(fullbin, (t_r, t_c), cands[0]) + cands
    select = select[:MAX_CANDIDATES]

    if _rows_mean_ok(select, t_r):
        chosen = select
    elif _rows_mean_ok(cands, t_r):
        # Mathematica's own fallback: the full (untruncated, unbridged) SortCoords chain.
        chosen = cands
    else:
        return [], decision_prob
    return [(r + r0 - pad, c + c0 - pad) for r, c in chosen], decision_prob
