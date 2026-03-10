"""Input/output utilities: CSV/JSON export, overlay visualization.

Handles track data export and kymograph overlay generation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from kymobutler.tracking import Track


def save_tracks_csv(tracks: list[Track], output_path: str | Path) -> None:
    """Save tracks to a CSV file.

    Format: track_id, time, space (one row per point).
    """
    path = Path(output_path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "time", "space"])
        for i, track in enumerate(tracks):
            for r, c in track.points:
                writer.writerow([i + 1, r, c])


def save_tracks_json(tracks: list[Track], output_path: str | Path) -> None:
    """Save tracks to a JSON file."""
    path = Path(output_path)
    data = {
        "tracks": [
            {"id": i + 1, "points": [{"time": r, "space": c} for r, c in track.points]}
            for i, track in enumerate(tracks)
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_statistics_csv(
    stats: list[dict], output_path: str | Path, pixel_time: float = 1.0, pixel_space: float = 1.0
) -> None:
    """Save track statistics to CSV."""
    path = Path(output_path)
    if not stats:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=stats[0].keys())
        writer.writeheader()
        writer.writerows(stats)


def create_overlay(
    image: np.ndarray,
    tracks: list[Track],
    output_path: str | Path | None = None,
) -> np.ndarray:
    """Create a colored track overlay on the kymograph image.

    Args:
        image: Grayscale kymograph (H, W) float32.
        tracks: List of tracks to overlay.
        output_path: Optional path to save the overlay as PNG.

    Returns:
        RGB overlay image (H, W, 3) as uint8.
    """
    from PIL import Image as PILImage

    h, w = image.shape[:2]

    # Convert grayscale to RGB
    if image.max() <= 1.0:
        gray_uint8 = (image * 255).astype(np.uint8)
    else:
        gray_uint8 = image.astype(np.uint8)
    rgb = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1)

    # Generate random colors for each track
    rng = np.random.default_rng(42)
    for track in tracks:
        color = rng.integers(50, 255, size=3).tolist()
        for i in range(len(track.points) - 1):
            r0, c0 = track.points[i]
            r1, c1 = track.points[i + 1]
            # Simple line drawing (Bresenham-like)
            _draw_line(rgb, int(r0), int(c0), int(r1), int(c1), color)

    if output_path is not None:
        pil_img = PILImage.fromarray(rgb)
        pil_img.save(output_path)

    return rgb


def _draw_line(
    img: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: list[int]
) -> None:
    """Draw a line on an RGB image using Bresenham's algorithm."""
    h, w = img.shape[:2]
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    while True:
        if 0 <= r0 < h and 0 <= c0 < w:
            img[r0, c0] = color
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc
