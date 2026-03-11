"""Particle tracking algorithms for kymograph analysis.

Corresponds to MakeTrack, GetNextCoord, CatchStraddlers, UniKymoButlerTrack,
BiKymoButlerTrack in KymoButler.wl.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from scipy.ndimage import label
from scipy.spatial import KDTree
from tqdm import tqdm

from kymobutler.config import (
    SEARCH_RADIUS,
    STRADDLER_MAX_ITERATIONS,
    STRADDLER_PIXEL_THRESHOLD,
)
from kymobutler.morphology import (
    detect_seeds,
    process_segmentation_bi,
    process_segmentation_uni,
    _filter_components,
)
from kymobutler.vision_module import get_candidates


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
    """Trim backwards-in-time portion from end of track.

    Corresponds to GoBack in KymoButler.wl lines 245-247.
    """
    i = -1
    while -i < len(track_points) and track_points[i][0] - track_points[i - 1][0] <= 0:
        i -= 1
    if i == -1:
        return track_points
    return track_points[:i]


def _make_track(
    kym: np.ndarray,
    allyx: np.ndarray,
    seed: tuple[int, int],
    kdtree: KDTree,
    vision_net: torch.nn.Module | None = None,
    vision_threshold: float = 0.5,
    device: str = "cpu",
) -> Track:
    """Build a track starting from a seed point using greedy nearest-neighbor extension.

    Corresponds to MakeTrack in KymoButler.wl lines 271-291.
    """
    h, w = kym.shape
    track = Track()
    decision_probs: list[float] = []

    # Find first neighbor
    indices = kdtree.query_ball_point(seed, r=SEARCH_RADIUS)
    neighbors = [tuple(allyx[i]) for i in indices if tuple(allyx[i]) != seed]

    if len(neighbors) == 0:
        track.points = [seed]
        return track
    if len(neighbors) > 1:
        # Pick the one with highest row (time) value
        neighbors.sort(key=lambda c: c[0], reverse=True)
    first = neighbors[0]

    track_points = [seed, first]
    backwards_count = 0
    visited = {seed, first}

    # Iteratively extend
    while True:
        last = track_points[-1]

        # Find candidates within search radius not already in track
        indices = kdtree.query_ball_point(last, r=SEARCH_RADIUS)
        candidates = [tuple(allyx[i]) for i in indices if tuple(allyx[i]) not in visited]

        if len(candidates) == 1:
            new_points = candidates
        elif len(candidates) != 1 and vision_net is not None and len(track_points) > 2:
            # Use vision module
            new_points, prob = get_candidates(
                kym, track_points, allyx, vision_threshold, vision_net, device
            )
            if prob > 0:
                decision_probs.append(prob)
        else:
            new_points = candidates if len(candidates) == 1 else []

        if len(new_points) == 0:
            break

        # Check for backwards movement
        last_row = track_points[-1][0]
        new_last_row = new_points[-1][0]

        if new_last_row > last_row:
            backwards_count = 0
        elif new_last_row < last_row:
            if backwards_count < 1:
                backwards_count += 1
            else:
                track_points = _go_back(track_points)
                break

        for p in new_points:
            visited.add(tuple(p) if not isinstance(p, tuple) else p)
        track_points.extend(
            [p if isinstance(p, tuple) else tuple(p) for p in new_points]
        )

    # Remove out-of-bounds coordinates (0-indexed)
    track_points = [
        (r, c) for r, c in track_points if 0 <= r < h and 0 <= c < w
    ]

    track.points = track_points
    track.decision_probs = decision_probs
    return track


def _average_duplicates(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Average multiple space values at the same timepoint.

    Corresponds to Round/@Mean/@GatherBy[#,First] in KymoButler.wl.
    """
    if not points:
        return points
    from collections import defaultdict

    by_time: dict[int, list[int]] = defaultdict(list)
    for r, c in points:
        by_time[r].append(c)
    result = [(r, round(np.mean(cols))) for r, cols in sorted(by_time.items())]
    return result


def _remove_subset_tracks(
    tracks: list[Track], min_size: int
) -> list[Track]:
    """Remove tracks that are subsets of other tracks.

    Corresponds to checkifAnyTrkisSubset in KymoButler.wl lines 471-473.
    """
    n = len(tracks)
    if n <= 1:
        return tracks

    point_sets = [set(map(tuple, t.points)) for t in tracks]
    keep = [True] * n

    for i in range(n):
        for j in range(n):
            if i == j or not keep[i] or not keep[j]:
                continue
            intersection = len(point_sets[i] & point_sets[j])
            if abs(intersection - len(point_sets[j])) <= min_size and len(point_sets[j]) < len(
                point_sets[i]
            ):
                keep[j] = False

    return [t for t, k in zip(tracks, keep) if k]


def _resolve_overlaps(tracks: list[Track]) -> list[Track]:
    """Resolve overlapping tracks by keeping the higher-confidence one.

    Corresponds to ovlpSegID / overlap resolution in KymoButler.wl lines 480-489.
    """
    if len(tracks) <= 1:
        return tracks

    changed = True
    while changed:
        changed = False
        n = len(tracks)
        point_sets = [set(map(tuple, t.points)) for t in tracks]

        for i in range(n):
            for j in range(i + 1, n):
                overlap = point_sets[i] & point_sets[j]
                if len(overlap) > 10:
                    # Keep the track with higher mean decision probability
                    prob_i = np.mean(tracks[i].decision_probs) if tracks[i].decision_probs else 0
                    prob_j = np.mean(tracks[j].decision_probs) if tracks[j].decision_probs else 0
                    if prob_i >= prob_j:
                        # Remove overlapping points from j
                        tracks[j].points = [p for p in tracks[j].points if tuple(p) not in point_sets[i]]
                    else:
                        tracks[i].points = [p for p in tracks[i].points if tuple(p) not in point_sets[j]]
                    changed = True
                    break
            if changed:
                break

    return [t for t in tracks if len(t.points) > 0]


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


def track_unidirectional(
    pred_dict: dict[str, np.ndarray],
    original_dims: tuple[int, int],
    threshold: float = 0.2,
    min_size: int = 3,
    min_frames: int = 3,
) -> tuple[list[Track], list[Track]]:
    """Track particles in a unidirectional kymograph.

    Uses connected component analysis on the anterograde and retrograde masks.
    No vision module needed.
    Corresponds to UniKymoButlerTrack in KymoButler.wl lines 60-97.

    Returns:
        (anterograde_tracks, retrograde_tracks)
    """

    def _extract_tracks(pred: np.ndarray) -> list[Track]:
        skeleton = process_segmentation_uni(pred, original_dims, threshold, min_size, min_frames)
        # Use 8-connectivity (matching Mathematica's MorphologicalComponents)
        structure = np.ones((3, 3), dtype=int)
        labeled, num = label(skeleton, structure=structure)
        tracks = []
        for i in range(1, num + 1):
            coords = list(zip(*np.where(labeled == i)))
            if not coords:
                continue
            coords = _average_duplicates(coords)
            # Filter by frame span
            if len(coords) >= 2 and coords[-1][0] - coords[0][0] >= min_frames:
                tracks.append(Track(points=coords))
        return tracks

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
    """Track particles in a bidirectional kymograph.

    Uses greedy nearest-neighbor tracking with vision module for ambiguity resolution.
    Corresponds to BiKymoButlerTrack in KymoButler.wl lines 426-523.
    """
    h, w = kym_preprocessed.shape
    original_dims = (h, w)

    # Phase 1: Skeleton preparation
    paths = process_segmentation_bi(prediction, original_dims, threshold, min_size, min_frames)

    # Get seeds and all foreground coordinates
    seeds = detect_seeds(paths)
    allyx_coords = np.array(list(zip(*np.where(paths))), dtype=np.float64)

    if len(seeds) == 0 or len(allyx_coords) == 0:
        return []

    # Build KDTree for spatial queries
    kdtree = KDTree(allyx_coords)

    # Phase 2: Initial track building
    tracks: list[Track] = []
    for seed in tqdm(seeds, desc="Building tracks", leave=False):
        trk = _make_track(
            kym_preprocessed, allyx_coords, seed, kdtree,
            vision_net, vision_threshold, device,
        )
        if len(trk.points) > 0:
            tracks.append(trk)

    # Phase 3: Straddler recovery
    tracked_pixels = set()
    for t in tracks:
        tracked_pixels.update(map(tuple, t.points))

    remaining = paths.copy()
    for r, c in tracked_pixels:
        ri, ci = round(r), round(c)
        if 0 <= ri < h and 0 <= ci < w:
            remaining[ri, ci] = False

    remaining = _filter_components(remaining, min_size=5, min_frames=3)

    iteration = 0
    while np.sum(remaining) > STRADDLER_PIXEL_THRESHOLD and iteration < STRADDLER_MAX_ITERATIONS:
        iteration += 1
        new_seeds = detect_seeds(remaining)
        if not new_seeds:
            break

        remaining_yx = np.array(list(zip(*np.where(remaining))), dtype=np.float64)
        if len(remaining_yx) == 0:
            break
        remaining_kdtree = KDTree(remaining_yx)

        for seed in new_seeds:
            trk = _make_track(
                kym_preprocessed, allyx_coords, seed, kdtree,
                vision_net, vision_threshold, device,
            )
            if len(trk.points) > 0:
                tracks.append(trk)
                for r, c in trk.points:
                    ri, ci = round(r), round(c)
                    if 0 <= ri < h and 0 <= ci < w:
                        remaining[ri, ci] = False

        remaining = _filter_components(remaining, min_size=5, min_frames=3)

    if len(tracks) == 0:
        return []

    # Phase 4: Post-tracking cleanup
    # Clamp coordinates to image bounds (0-indexed)
    for track in tracks:
        track.points = [
            (max(0, min(r, h - 1)), max(0, min(c, w - 1))) for r, c in track.points
        ]

    # Average duplicates at same timepoint
    for track in tracks:
        track.points = _average_duplicates(track.points)

    # Remove subset tracks
    tracks = _remove_subset_tracks(tracks, min_size)

    # Resolve overlaps
    tracks = _resolve_overlaps(tracks)

    # Remove subsets again
    tracks = _remove_subset_tracks(tracks, min_size)

    # Split at large gaps
    split_tracks: list[Track] = []
    for track in tracks:
        split_tracks.extend(_split_at_gaps(track, max_gap=2 * min_size))
    tracks = split_tracks

    # Filter by minimum frame span
    tracks = [
        t for t in tracks if len(t.points) >= 2 and t.points[-1][0] - t.points[0][0] >= min_frames
    ]

    return tracks
