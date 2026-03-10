"""Benchmarking: precision, recall, F1 for track predictions.

Corresponds to benchmarkPrediction in KymoButler.wl lines 540-564.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from kymobutler.tracking import Track


def _find_close_tracks(
    nearest_func: KDTree, track_points: list[tuple[int, int]], radius: float = 3.2
) -> list[tuple[int, int]]:
    """Find all points from a reference set within radius of any point in track."""
    if len(track_points) == 0:
        return []
    indices = set()
    for point in track_points:
        nearby = nearest_func.query_ball_point(point, r=radius)
        indices.update(nearby)
    return list(indices)


def _recall_single(
    pred_kdtrees: list[KDTree], gt_track: Track, radius: float = 3.2
) -> float:
    """Compute recall for a single ground-truth track against all predicted tracks.

    Finds the predicted track with the most matching points and computes:
    1 - |len(matched) - len(gt)| / len(gt)
    """
    gt_points = gt_track.points
    if not gt_points:
        return 0.0

    best_match_len = 0
    for kdtree in pred_kdtrees:
        matched = _find_close_tracks(kdtree, gt_points, radius)
        if len(matched) > best_match_len:
            best_match_len = len(matched)

    if best_match_len == 0:
        return 0.0
    return max(0.0, 1.0 - abs(best_match_len - len(gt_points)) / len(gt_points))


def _precision_single(
    gt_kdtrees: list[KDTree], pred_track: Track, radius: float = 3.2
) -> float:
    """Compute precision for a single predicted track against all ground-truth tracks."""
    pred_points = pred_track.points
    if not pred_points:
        return 0.0

    best_match_len = 0
    for kdtree in gt_kdtrees:
        matched = _find_close_tracks(kdtree, pred_points, radius)
        if len(matched) > best_match_len:
            best_match_len = len(matched)

    if best_match_len == 0:
        return 0.0
    return max(0.0, 1.0 - abs(best_match_len - len(pred_points)) / len(pred_points))


def benchmark_prediction(
    pred_tracks: list[Track],
    gt_tracks: list[Track],
    radius: float = 3.2,
) -> dict[str, float]:
    """Compute precision, recall, and F1 score between predicted and ground-truth tracks.

    Corresponds to benchmarkPrediction in KymoButler.wl lines 556-564.

    Args:
        pred_tracks: Predicted tracks.
        gt_tracks: Ground-truth tracks.
        radius: Matching radius in pixels.

    Returns:
        Dict with 'recall', 'precision', 'f1' keys.
    """
    # Build KDTrees for each track
    pred_kdtrees = []
    for t in pred_tracks:
        if t.points:
            pred_kdtrees.append(KDTree(np.array(t.points, dtype=np.float64)))

    gt_kdtrees = []
    for t in gt_tracks:
        if t.points:
            gt_kdtrees.append(KDTree(np.array(t.points, dtype=np.float64)))

    # Recall: mean over GT tracks
    if gt_tracks:
        recall = float(np.mean([_recall_single(pred_kdtrees, t, radius) for t in gt_tracks]))
    else:
        recall = 0.0

    # Precision: mean over predicted tracks
    if pred_tracks:
        precision = float(
            np.mean([_precision_single(gt_kdtrees, t, radius) for t in pred_tracks])
        )
    else:
        precision = 0.0

    # F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {"recall": recall, "precision": precision, "f1": f1}
