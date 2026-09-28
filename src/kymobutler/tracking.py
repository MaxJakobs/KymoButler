"""Particle tracking algorithms for kymograph analysis.

Corresponds to MakeTrack, GetNextCoord, CatchStraddlers, UniKymoButlerTrack,
BiKymoButlerTrack in KymoButler.wl.

Coordinates are 0-based (row, col) = (time, space). Mathematica is 1-based; the only step
whose result depends on the base is the rounding in ``Round/@Mean/@GatherBy[#, First]``,
which ``_gather_round_mean`` therefore performs on 1-based columns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
import torch
from scipy.ndimage import binary_dilation
from scipy.sparse import csr_matrix
from tqdm import tqdm

from kymobutler.config import (
    STRADDLER_MAX_ITERATIONS,
    STRADDLER_PIXEL_THRESHOLD,
)
from kymobutler.morphology import (
    _filter_components,
    detect_seeds,
    morphological_components,
    process_segmentation_bi,
    process_segmentation_uni,
)
from kymobutler.vision_module import VisionContext, get_candidates

logger = logging.getLogger(__name__)

_NEIGHBOURS_8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]
# Track coordinates can leave the image by up to 25 px (vision-module tiles reach into the
# padding of GetCand); the neighbour lookup pads the skeleton by more than that.
_LOOKUP_PAD = 27

Point = tuple[int, int]


@dataclass
class Track:
    """A single particle track."""

    points: list[tuple[int, int]] = field(default_factory=list)
    decision_probs: list[float] = field(default_factory=list)

    @property
    def duration(self) -> int:
        if len(self.points) < 2:
            return 0
        return self.points[-1][0] - self.points[0][0]

    @property
    def direction(self) -> int:
        if len(self.points) < 2:
            return 0
        return int(np.sign(self.points[-1][1] - self.points[0][1]))


def _go_back(track_points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Trim the backwards-in-time end of a track.

    GoBack in KymoButler.wl lines 245-247::

        While[-i < Length@x && x[[i,1]] - x[[i-1,1]] <= 0, i--]; Drop[x, i]

    ``Drop[x, i]`` drops the last ``-i`` points, so the last point is always dropped (also
    when the loop does not run, ``i == -1``).
    """
    i = -1
    while -i < len(track_points) and track_points[i][0] - track_points[i - 1][0] <= 0:
        i -= 1
    return track_points[:i]


class _Skeleton:
    """The bidirectional skeleton ``paths`` with O(1) 8-neighbour lookups."""

    def __init__(self, paths: np.ndarray):
        self.h, self.w = paths.shape
        self.padded = np.pad(paths.astype(bool), _LOOKUP_PAD)

    def neighbours(self, p: Point) -> list[Point]:
        """Skeleton pixels within distance 1.5 of ``p``, excluding ``p`` itself.

        ``Nearest[allyx, p, {All, 1.5}]`` minus ``p``: for integer coordinates these are
        exactly the 8-neighbours.
        """
        r, c = p
        pr, pc = r + _LOOKUP_PAD, c + _LOOKUP_PAD
        return [(r + dr, c + dc) for dr, dc in _NEIGHBOURS_8 if self.padded[pr + dr, pc + dc]]


def _make_track(
    skel: _Skeleton,
    ctx: VisionContext | None,
    seed: Point,
) -> tuple[list[Point], list[float] | None]:
    """Build a track from a seed (MakeTrack + GetNextCoord, KymoButler.wl lines 249-291).

    Returns:
        (points, sown): the in-image track points, and the values this call Sows to
        "DecisionProb": the decision probability of every vision-module call that found a
        component, followed by the ``Sow[1]`` MakeTrack adds when the track is done. ``sown``
        is None when the seed has no skeleton neighbour: that branch of MakeTrack returns
        ``{seed}`` without Sowing anything.

    Raises:
        RuntimeError: if the track goes round in a loop (its state repeats, so Mathematica's
            NestWhile would never end), or, as a last resort, if it needs more than
            ``_max_vision_calls`` decision-module calls.
    """
    h, w = skel.h, skel.w
    nbrs = skel.neighbours(seed)
    if not nbrs:
        return [seed], None
    # If[Length@tmp > 1, tmp = {Last@SortBy[tmp, First]}]: largest row, ties by canonical
    # order, i.e. largest column.
    first = max(nbrs)

    track = [seed, first]
    in_track = {seed, first}
    sown: list[float] = []
    backwards_count = 0
    # Non-termination guard. Mathematica's NestWhile has no bound: MakeTrack never ends if its
    # state repeats. A GetNextCoord step reads only the set S of track points (the candidates'
    # DeleteCases), the last point (candidates; GetCand's Nearest), the second-to-last point
    # (GetCand drops the last point: tile centre, lastTrk), the set of the track without its
    # last point (GetCand's track image and Complement: S, or S minus the last point if that
    # occurs only once), backwards_count and Length@trk > 2 (true once the track has grown;
    # GoBack reads the whole track, but ends it). The step is deterministic, so the track
    # never ends iff (S, last, second-to-last, last-occurs-earlier, backwards_count) repeats.
    # S only grows: the other four are remembered while |S| stays the same. Every track point
    # lies in the padded image, so there are finitely many states and a track either ends or
    # repeats one; the call budget only caps the time a pathological track may take.
    epoch = len(in_track)
    seen: set[tuple] = set()
    last_repeated = False
    vision_calls = 0
    max_vision_calls = _max_vision_calls(h, w)
    while True:
        last = track[-1]
        cand = [p for p in skel.neighbours(last) if p not in in_track]
        if len(cand) == 1:
            new = cand
        elif len(track) > 2 and ctx is not None:
            vision_calls += 1
            if vision_calls > max_vision_calls:
                raise RuntimeError(
                    f"KymoButler stopped tracing the track that starts at {_where(seed, h, w)} "
                    f"({_POSITION_NOTE}): it had not ended at {_where(last, h, w)} after "
                    f"{max_vision_calls} decision-module steps ({_BUDGET_NOTE}), far more than "
                    "real tracks need. Please report this image; a different threshold or "
                    "vision_threshold may avoid it."
                )
            new, prob = get_candidates(ctx, track)
            if prob is not None:
                sown.append(prob)
        else:
            new = []

        if new:
            step = new[-1][0] - last[0]
            if step > 0:
                backwards_count = 0
            elif step < 0:
                if backwards_count < 1:
                    backwards_count += 1
                else:
                    track = _go_back(track)
                    new = []
        if not new:
            break
        if new[-1] == _END_MARKER:
            # NestWhile[..., Last@First@# != {0,0} &] also stops at a genuine point {0,0} (just
            # outside the top-left corner, 1-based) that the decision module returns last, and
            # Most@ then drops that point as if it were the end marker.
            track.extend(new[:-1])
            break
        last_repeated = new[-1] in in_track or new[-1] in new[:-1]
        track.extend(new)
        in_track.update(new)
        if len(in_track) != epoch:
            epoch = len(in_track)
            seen.clear()
        state = (track[-1], track[-2], last_repeated, backwards_count)
        if state in seen:
            raise RuntimeError(
                f"KymoButler cannot finish the track that starts at {_where(seed, h, w)} "
                f"({_POSITION_NOTE}): from {_where(track[-1], h, w)} on, the decision module "
                "keeps sending the track back over points it already has, so the original "
                "(Mathematica) KymoButler would never finish this kymograph either. A different "
                "threshold or vision_threshold may avoid this loop."
            )
        seen.add(state)

    sown.append(1.0)  # Sow[1, "DecisionProb"]; Sow["TrkDone", "DecisionProb"]
    return [(r, c) for r, c in track if 0 <= r < h and 0 <= c < w], sown


def _max_vision_calls(h: int, w: int) -> int:
    """Last-resort cap on the decision-module calls of one MakeTrack: ``4 * (H + W)``, at least
    1000.

    Not needed for termination (the state-repeat test in ``_make_track`` is exact); it caps the
    time a pathological, still-growing track may take: at about 13 ms per call, 2.3 min on a
    1500 x 1200 kymograph, 13 s on a small one. Real tracks need far fewer calls (see
    ``track_paths_bidirectional``).
    """
    return max(4 * (h + w), _MIN_VISION_BUDGET)


_MIN_VISION_BUDGET = 1000
_BUDGET_NOTE = f"4 x (height + width) of this kymograph, at least {_MIN_VISION_BUDGET}"
# MakeTrack's end-of-track marker {0,0} (1-based), as a 0-based point
_END_MARKER = (-1, -1)


_POSITION_NOTE = (
    "pixels from the top-left corner, as ImageJ/Fiji counts them: x is space, y is time"
)


def _where(p: Point, h: int, w: int) -> str:
    """A (row, col) track point of an H x W kymograph for error messages (``_POSITION_NOTE``)."""
    r, c = p
    inside = 0 <= r < h and 0 <= c < w
    return f"x = {c}, y = {r}" + ("" if inside else " (just outside the image)")


def _straddler_mask(paths: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """pathstmp = SelectComponents[paths - tracks, #Count > 5 && span >= 3 &] (KymoButler.wl 444)."""
    return _filter_components(paths & ~covered, min_size=6, min_frames=3)


def _straddler_seeds(pathstmp: np.ndarray) -> list[Point]:
    """Seeds for the left-over skeleton pieces (CatchStraddlers, KymoButler.wl lines 395-412).

    Pieces are grouped by the connected components of ``Dilation[pathstmp, 3]`` (7x7 box).
    Each group is keyed by its topmost-leftmost pixel. A group whose key lies above all its
    seeds is seeded at the key; otherwise at its topmost seeds; these groups come in the order
    of their first seed. Groups without any seed are seeded at their key, after the others, in
    the order of the dilated components' MorphologicalComponents labels (the order of
    ComponentMeasurements' masks), which is not their raster order.
    """
    dilated, _ = morphological_components(
        binary_dilation(pathstmp, structure=np.ones((7, 7), dtype=bool))
    )
    w = pathstmp.shape[1]
    flat = np.flatnonzero(pathstmp)  # row-major = ArrayRules order
    group_of_pixel = dilated.ravel()[flat]
    groups, first_idx = np.unique(group_of_pixel, return_index=True)
    key_of_group = {int(g): divmod(int(flat[i]), w) for g, i in zip(groups, first_idx)}

    seeds = [(int(r), int(c)) for r, c in detect_seeds(pathstmp)]  # sorted by (row, col)
    by_key: dict[Point, list[Point]] = {}
    for s in seeds:
        by_key.setdefault(key_of_group[int(dilated[s])], []).append(s)

    out: list[Point] = []
    for key, group_seeds in by_key.items():
        top = min(s[0] for s in group_seeds)
        out += [key] if key[0] < top else [s for s in group_seeds if s[0] == top]
    out += [key for g, key in sorted(key_of_group.items()) if key not in by_key]
    return out


def _gather_round_mean(points: list[Point]) -> list[Point]:
    """``Round/@Mean/@GatherBy[trk, First]`` (KymoButler.wl line 463).

    One point per time, in order of first appearance, at the mean column rounded half to
    even. Mathematica rounds the exact mean of the 1-based columns, so the rounding is done
    there (a 0-based rounding differs by one pixel whenever the mean ends in .5).
    """
    by_time: dict[int, list[int]] = {}
    for r, c in points:
        by_time.setdefault(r, []).append(c + 1)
    out = []
    for r, cols in by_time.items():
        col = cols[0] if len(cols) == 1 else round(Fraction(sum(cols), len(cols)))
        out.append((r, col - 1))
    return out


def _uni_tracks(skeleton: np.ndarray, min_frames: int) -> list[list[Point]]:
    """Tracks of a filtered unidirectional skeleton (UniKymoButlerTrack, KymoButler.wl 64-79)::

        tmpa = MorphologicalComponents@tmpa
        antrks = Transpose[{dim[[2]]+1-PixelValuePositions[antPaths,#][[;;,2]],
                            PixelValuePositions[antPaths,#][[;;,1]]}]& /@ Range[...]
        antrks = Map[Round/@Mean/@GatherBy[#, First]&, antrks]
        antrks = Select[antrks, First@Last@# - First@First@# >= minFr &]

    One track per component, in MorphologicalComponents order (``morphological_components``).
    PixelValuePositions lists pixels row by row from the top, so every track is sorted by
    time; each time gets the mean column rounded half to even in 1-based coordinates.
    Returns 0-based (time, space) points.
    """
    labels, num = morphological_components(skeleton)
    if num == 0:
        return []
    rr, cc = np.nonzero(labels)
    lab = labels[rr, cc]
    order = np.lexsort((rr, lab))  # by component, then time (stable: columns stay ascending)
    lab, rr, cc = lab[order], rr[order], cc[order]
    key = lab * (skeleton.shape[0] + 1) + rr
    _, start, count = np.unique(key, return_index=True, return_counts=True)
    # exact mean of the 1-based columns, rounded half to even (np.round of the correctly
    # rounded quotient: a .5 quotient is exact in binary, any other is >= 1/(2n) away from it)
    col1 = np.round(np.add.reduceat(cc + 1, start) / count).astype(np.int64)
    t_lab, t_row, t_col = lab[start], rr[start], col1 - 1
    bounds = np.searchsorted(t_lab, np.arange(1, num + 2))
    tracks: list[list[Point]] = []
    for i in range(num):
        lo, hi = bounds[i], bounds[i + 1]
        if t_row[hi - 1] - t_row[lo] >= min_frames:
            tracks.append(list(zip(t_row[lo:hi].tolist(), t_col[lo:hi].tolist())))
    return tracks


def _intersections(tracks: list[list[Point]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(i, j, |T_i ∩ T_j|) for every ordered pair i != j of tracks sharing a point."""
    index: dict[Point, int] = {}
    rows: list[int] = []
    cols: list[int] = []
    for i, t in enumerate(tracks):
        for p in set(t):
            cols.append(index.setdefault(p, len(index)))
            rows.append(i)
    if not rows:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    a = csr_matrix(
        (np.ones(len(rows), dtype=np.int64), (rows, cols)), shape=(len(tracks), len(index))
    )
    c = (a @ a.T).tocoo()
    off = c.row != c.col
    return c.row[off], c.col[off], c.data[off]


def _subset_keep(tracks: list[list[Point]], min_size: int) -> np.ndarray:
    """Which tracks survive ``checkifAnyTrkisSubset`` (KymoButler.wl lines 471-473, 491-493).

    Track j is dropped if, for some other track i, ``Abs[|T_i ∩ T_j| - Length[T_j]] <= minSz
    && Length[T_j] < Length[T_i]``. All tracks are tested against the same list and removed
    together (``sel = Nor@@@...``). Note that the test holds for every track of at most
    ``minSz`` points whenever a longer track exists, overlapping or not.
    """
    n = len(tracks)
    keep = np.ones(n, dtype=bool)
    if n <= 1:
        return keep
    lengths = np.array([len(t) for t in tracks], dtype=np.int64)
    # pairs without a common point: |0 - L_j| <= minSz && L_j < L_i
    keep &= ~((lengths <= min_size) & (lengths < lengths.max()))
    i, j, inter = _intersections(tracks)
    hit = (np.abs(inter - lengths[j]) <= min_size) & (lengths[j] < lengths[i])
    keep[j[hit]] = False
    return keep


def _overlap_groups(tracks: list[list[Point]]) -> list[list[int]]:
    """ovlpIDs (KymoButler.wl lines 480-481): for each track i (in order) that shares more
    than 10 points with a later track, the group ``{i, j1, j2, ...}`` of those later tracks."""
    i, j, inter = _intersections(tracks)
    sel = (j > i) & (inter > 10)
    groups: dict[int, list[int]] = {}
    for a, b in sorted(zip(i[sel].tolist(), j[sel].tolist())):
        groups.setdefault(a, [a]).append(b)
    return [groups[a] for a in sorted(groups)]


def _resolve_overlaps(tracks: list[list[Point]], ptrk: list[float]) -> list[list[Point]]:
    """Resolve overlapping tracks (KymoButler.wl lines 479-489).

    For every group (swept in order, each seeing the tracks as modified by the previous
    ones) the member with the highest mean decision probability (the first one on ties)
    wins and every other member loses the winner's points. Groups are recomputed and swept
    again until none is left. Emptied tracks are kept (the next subset pass removes them).
    """
    tracks = [list(t) for t in tracks]
    groups = _overlap_groups(tracks)
    while groups:
        for group in groups:
            probs = [ptrk[k] for k in group]
            winner = group[int(np.argmax(probs))]
            won = set(tracks[winner])
            for k in group:
                if k != winner:
                    tracks[k] = [p for p in tracks[k] if p not in won]
        groups = _overlap_groups(tracks)
    return tracks


def _split_at_gaps(
    track: Track, max_gap: int
) -> list[Track]:
    """Split a track at large temporal gaps.

    Corresponds to Split[#, #2[[1]]-#1[[1]] <= 2*minSz &] in KymoButler.wl line 503.
    """
    if len(track.points) < 2:
        return [track]

    segments: list[list[tuple[int, int]]] = [[track.points[0]]]
    for i in range(1, len(track.points)):
        gap = track.points[i][0] - track.points[i - 1][0]
        if gap > max_gap:
            segments.append([track.points[i]])
        else:
            segments[-1].append(track.points[i])

    return [Track(points=seg, decision_probs=track.decision_probs) for seg in segments]


def _clean_up(
    tracks: list[list[Point]],
    sown: list[list[float] | None],
    shape: tuple[int, int],
    min_size: int,
    min_frames: int,
) -> list[Track]:
    """BiKymoButlerTrack's clean-up of the raw tracks (KymoButler.wl lines 454-506).

    Args:
        tracks: Raw tracks from MakeTrack, in order.
        sown: Per track, the values it Sowed to "DecisionProb" (None: nothing Sown).
        shape: (H, W) of the kymograph.
    """
    if not tracks:
        return []
    h, w = shape
    # ptrk: mean of every track's Sown run. A no-neighbour seed Sows nothing, so in Mathematica
    # the runs no longer line up with the tracks, Pick[ptrk, sel] fails (Pick::incomp) and ptrk
    # is garbage; that only matters if there are overlaps to resolve (checked below). Such a
    # seed is a one-pixel component of paths (CatchStraddlers' seeds lie in pieces of 6+
    # pixels), and a single pixel (Count 1, BoundingBox 1 row tall) passes SelectComponents
    # iff min_size <= 1 and min_frames <= 1 (Mathematica 14.3): with min_size > 1 or
    # min_frames > 1 this cannot happen.
    ptrk = [None if s is None else sum(s) / len(s) for s in sown]
    probs_of = [[] if s is None else s[:-1] for s in sown]  # vision-module probabilities
    misaligned = any(s is None for s in sown)

    tracks = [[(min(max(r, 0), h - 1), min(max(c, 0), w - 1)) for r, c in t] for t in tracks]
    tracks = [_gather_round_mean(t) for t in tracks]

    def pick(keep: np.ndarray) -> None:
        nonlocal tracks, ptrk, probs_of
        tracks = [t for t, k in zip(tracks, keep) if k]
        ptrk = [p for p, k in zip(ptrk, keep) if k]
        probs_of = [p for p, k in zip(probs_of, keep) if k]

    pick(_subset_keep(tracks, min_size))
    if misaligned and _overlap_groups(tracks):
        if min_size > 1 or min_frames > 1:  # a skeleton not filtered by process_segmentation_bi
            raise ValueError(
                "track_paths_bidirectional was given a skeleton with one-pixel pieces, which "
                f"process_segmentation_bi removes with min_size = {min_size} and min_frames = "
                f"{min_frames}; on a kymograph with overlapping tracks BiKymoButlerTrack has no "
                "defined result for them. Build the skeleton with process_segmentation_bi (or "
                "call track_bidirectional)."
            )
        raise ValueError(
            f"KymoButler cannot analyse this kymograph with min_size = {min_size} and "
            f"min_frames = {min_frames}: with both at 1 or less, one-pixel specks of the "
            "detected skeleton count as tracks, and on a kymograph with overlapping tracks, "
            "like this one, the original (Mathematica) KymoButler then has no defined result. "
            "Set min_size (minimum object size, in pixels) or min_frames (minimum frames per "
            "track) to 2 or more."
        )
    tracks = _resolve_overlaps(tracks, ptrk)
    pick(_subset_keep(tracks, min_size))

    # Split tracks at gaps > 2*minSz, keep those spanning >= minFr frames (last - first).
    out: list[Track] = []
    for t, probs in zip(tracks, probs_of):
        for seg in _split_at_gaps(Track(points=t, decision_probs=list(probs)), 2 * min_size):
            if seg.points and seg.points[-1][0] - seg.points[0][0] >= min_frames:
                out.append(seg)
    return out


def track_unidirectional(
    pred_dict: dict[str, np.ndarray],
    original_dims: tuple[int, int],
    threshold: float = 0.2,
    min_size: int = 3,
    min_frames: int = 3,
) -> tuple[list[Track], list[Track]]:
    """Track particles in a unidirectional kymograph (UniKymoButlerTrack, KymoButler.wl 60-97).

    One track per connected component of the filtered anterograde and retrograde skeletons
    (``process_segmentation_uni``); no vision module.

    Args:
        pred_dict: 'ant' and 'ret' probability maps from ``segment_unidirectional`` (network
            resolution; resized to the kymograph after binarization).
        original_dims: (height, width) of the kymograph.

    Returns:
        (anterograde_tracks, retrograde_tracks), each in Mathematica's order (its
        MorphologicalComponents numbering, which also numbers the tracks in the outputs).

    Raises:
        ValueError: for network-resolution maps of a kymograph smaller than 9 x 9 px, which
            Mathematica cannot segment (``morphology.check_image_size``).
    """

    def _extract_tracks(pred: np.ndarray) -> list[Track]:
        skeleton = process_segmentation_uni(pred, original_dims, threshold, min_size, min_frames)
        return [Track(points=pts) for pts in _uni_tracks(skeleton, min_frames)]

    ant_tracks = _extract_tracks(pred_dict["ant"])
    ret_tracks = _extract_tracks(pred_dict["ret"])

    return ant_tracks, ret_tracks


def track_bidirectional(
    prediction: np.ndarray,
    kym_preprocessed: np.ndarray,
    was_negated: bool,
    threshold: float = 0.2,
    vision_threshold: float = 0.5,
    vision_net: torch.nn.Module | None = None,
    min_size: int = 10,
    min_frames: int = 10,
    device: str = "cpu",
) -> list[Track]:
    """Track particles in a bidirectional kymograph (BiKymoButlerTrack, KymoButler.wl 426-523).

    Builds the skeleton (``process_segmentation_bi``) and tracks along it
    (``track_paths_bidirectional``).

    Args:
        prediction: Foreground probability map from ``segment_bidirectional`` (network
            resolution; resized to the kymograph after binarization).
        kym_preprocessed: Preprocessed kymograph (H, W); defines the output dimensions.
        vision_net: The decision module. With None, tracks stop wherever the skeleton is
            ambiguous (Mathematica always has one).

    Raises:
        ValueError: for a network-resolution prediction of a kymograph smaller than 9 x 9 px
            (before any tracking, ``morphology.check_image_size``); with ``min_size <= 1`` and
            ``min_frames <= 1``, if one-pixel skeleton pieces survive and the kymograph has
            overlapping tracks: Mathematica's result is undefined there (``Pick::incomp``, then
            ``Extract::partw``). Impossible with ``min_size > 1`` or ``min_frames > 1``.
        RuntimeError: if a track would never end (see ``track_paths_bidirectional``).
    """
    paths = process_segmentation_bi(
        prediction, kym_preprocessed.shape, threshold, min_size, min_frames
    )
    return track_paths_bidirectional(
        paths, kym_preprocessed, vision_threshold, vision_net, min_size, min_frames, device
    )


def track_paths_bidirectional(
    paths: np.ndarray,
    kym_preprocessed: np.ndarray,
    vision_threshold: float = 0.5,
    vision_net: torch.nn.Module | None = None,
    min_size: int = 10,
    min_frames: int = 10,
    device: str = "cpu",
) -> list[Track]:
    """BiKymoButlerTrack from its skeleton ``paths`` on (KymoButler.wl lines 433-506).

    Greedy tracking along the skeleton, with the vision (decision) module resolving
    ambiguities, then straddler recovery and Mathematica's clean-up passes.

    Args:
        paths: The filtered bidirectional skeleton (H, W), as from ``process_segmentation_bi``.
        kym_preprocessed: Preprocessed kymograph (H, W), the vision module's input.

    Raises:
        ValueError: if ``paths`` has one-pixel pieces and the kymograph overlapping tracks
            (see ``track_bidirectional``).
        RuntimeError: if a track goes round in a loop, i.e. MakeTrack's state repeats and
            Mathematica's NestWhile would never end (detected exactly, at the first repeat),
            or, as a last resort, if one track needs more than ``4 * (H + W)`` decision-module
            calls. Neither happens on any validation kymograph: the 721/721 Mathematica-referenced
            tracks (14 lab cases, 3 public, 4 synthetic up to 852 x 852 px) and 276 full
            pipeline runs on public and dense synthetic kymographs (18402 tracks) need at most
            23 calls per track, 0.05 * (H + W).
    """
    paths = np.asarray(paths, dtype=bool)
    h, w = kym_preprocessed.shape
    if paths.shape != (h, w):
        raise ValueError(f"skeleton shape {paths.shape} != kymograph shape {(h, w)}")
    if not paths.any():
        return []
    skel = _Skeleton(paths)
    ctx = (
        VisionContext(kym_preprocessed, paths, vision_net, vision_threshold, device)
        if vision_net is not None
        else None
    )

    tracks: list[list[Point]] = []
    sown: list[list[float] | None] = []
    covered = np.zeros((h, w), dtype=bool)

    def run(seeds: list[Point], desc: str) -> None:
        for seed in tqdm(seeds, desc=desc, leave=False):
            pts, s = _make_track(skel, ctx, seed)
            tracks.append(pts)
            sown.append(s)
            for r, c in pts:
                covered[r, c] = True

    # Phase 2: a track from every seed (KymoButler.wl 440-444)
    run([(int(r), int(c)) for r, c in detect_seeds(paths)], "Building tracks")

    # Phase 3: straddlers (KymoButler.wl 444-452). The loop runs at most 499 times.
    pathstmp = _straddler_mask(paths, covered)
    iteration = 1
    while pathstmp.sum() > STRADDLER_PIXEL_THRESHOLD:
        iteration += 1
        if iteration > STRADDLER_MAX_ITERATIONS:
            logger.warning(
                "Straddler recovery stopped after %d rounds with %d skeleton pixels left "
                "(Mathematica: 'Warning! Inf Loop Abort!!')",
                STRADDLER_MAX_ITERATIONS - 1,
                int(pathstmp.sum()),
            )
            break
        run(_straddler_seeds(pathstmp), "Catching straddlers")
        pathstmp = _straddler_mask(paths, covered)

    # Phase 4: clean-up (KymoButler.wl 456-506)
    return _clean_up(tracks, sown, (h, w), min_size, min_frames)
