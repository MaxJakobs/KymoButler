"""Post-processing: extract kinetic parameters from tracks.

Corresponds to KymoButlerPProc.wl.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kymobutler.tracking import Track


@dataclass
class TrackStatistics:
    """Kinetic parameters for a single track."""

    direction: int  # +1 (anterograde) or -1 (retrograde), 0 if stationary
    mean_velocity: float  # mean frame-to-frame velocity (pixels/frame)
    total_distance: float  # sum of absolute displacements (pixels)
    duration: int  # number of frames
    start_to_end_velocity: float  # total_distance / duration (pixels/frame)


def get_derived_quantities(track: Track) -> TrackStatistics | None:
    """Compute kinetic parameters from a single track.

    Corresponds to getDerivedQuantities in KymoButlerPProc.wl lines 16-23.

    The velocity calculation matches Mathematica's:
        v = Mean[Flatten[Abs/@Ratios/@Differences@trk]]
    This computes element-wise ratios of consecutive difference vectors.
    """
    points = np.array(track.points, dtype=np.float64)
    if len(points) < 2:
        return None

    diffs = np.diff(points, axis=0)  # (N-1, 2)

    # Velocity: Mean of absolute ratios of consecutive differences
    if len(diffs) >= 2:
        # Avoid division by zero
        safe_diffs = np.where(diffs[:-1] != 0, diffs[:-1], 1.0)
        ratios = diffs[1:] / safe_diffs
        velocity = float(np.mean(np.abs(ratios)))
    else:
        velocity = 0.0

    # Direction: sign of net spatial displacement
    direction = int(np.sign(points[-1, 1] - points[0, 1]))

    # Total distance: sum of absolute spatial displacements
    total_distance = float(np.sum(np.abs(diffs[:, 1])))

    # Duration in frames
    duration = int(abs(points[-1, 0] - points[0, 0])) + 1

    # Start-to-end velocity
    s2e_velocity = total_distance / duration if duration > 0 else 0.0

    return TrackStatistics(
        direction=direction,
        mean_velocity=velocity,
        total_distance=total_distance,
        duration=duration,
        start_to_end_velocity=s2e_velocity,
    )


def postprocess(
    tracks: list[Track],
    pixel_size_time: float = 1.0,
    pixel_size_space: float = 1.0,
) -> list[dict]:
    """Compute statistics for all tracks with physical unit scaling.

    Args:
        tracks: List of Track objects.
        pixel_size_time: Seconds per pixel in the time dimension.
        pixel_size_space: Micrometers per pixel in the space dimension.

    Returns:
        List of dicts with keys: direction, velocity_um_per_sec, duration_sec,
        distance_um, start2end_velocity_um_per_sec.
    """
    results = []
    for track in tracks:
        stats = get_derived_quantities(track)
        if stats is None:
            continue
        results.append(
            {
                "direction": stats.direction,
                "velocity_um_per_sec": round(
                    pixel_size_space / pixel_size_time * stats.mean_velocity, 4
                ),
                "duration_sec": round(pixel_size_time * stats.duration, 4),
                "distance_um": round(pixel_size_space * stats.total_distance, 4),
                "start2end_velocity_um_per_sec": round(
                    pixel_size_space * stats.start_to_end_velocity / pixel_size_time, 4
                ),
            }
        )
    return results
