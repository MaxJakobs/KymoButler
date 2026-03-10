"""Vision module inference wrapper and tile extraction.

Corresponds to GetTile, GetCandFromPmap, GetCand in KymoButler.wl lines 148-243.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import label
from scipy.spatial import KDTree

from kymobutler.config import VISION_MODULE_TILE_SIZE, MAX_CANDIDATES
from kymobutler.graph_utils import find_shortest_path_on_skeleton


def get_tile(
    kym: np.ndarray,
    track: list[tuple[int, int]],
    allyx: np.ndarray,
    tile_size: int = VISION_MODULE_TILE_SIZE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]]]:
    """Extract a tile centered on the last track point.

    Corresponds to GetTile in KymoButler.wl lines 148-162.

    Args:
        kym: Full kymograph image (H, W) float32.
        track: Track coordinates as list of (row, col).
        allyx: All skeleton coordinates (N, 2) array.
        tile_size: Size of the extracted tile.

    Returns:
        (tile_image, track_mask, structure_mask, window)
        - tile_image: (tile_size, tile_size) kymograph crop
        - track_mask: (tile_size, tile_size) binary mask of track points in tile
        - structure_mask: (tile_size, tile_size) binary mask of all skeleton in tile
        - window: [[row_start, row_end], [col_start, col_end]]
    """
    h, w = kym.shape
    last = track[-1]
    half = tile_size // 2

    # Compute window
    row_start = round(last[0] - half)
    row_end = round(last[0] + half - 1)
    col_start = round(last[1] - half)
    col_end = round(last[1] + half - 1)

    # Boundary correction (shift window to stay within image)
    if row_start < 0:
        row_end -= row_start
        row_start = 0
    elif row_end >= h:
        row_start -= (row_end - h + 1)
        row_end = h - 1

    if col_start < 0:
        col_end -= col_start
        col_start = 0
    elif col_end >= w:
        col_start -= (col_end - w + 1)
        col_end = w - 1

    # Clamp to valid range
    row_start = max(0, row_start)
    col_start = max(0, col_start)
    row_end = min(h - 1, row_end)
    col_end = min(w - 1, col_end)

    win = [[row_start, row_end + 1], [col_start, col_end + 1]]

    # Extract tile
    tile = kym[win[0][0]:win[0][1], win[1][0]:win[1][1]]
    # Rescale to [0, 1]
    if tile.max() > tile.min():
        tile = (tile - tile.min()) / (tile.max() - tile.min())

    # Track mask
    track_mask = np.zeros((h, w), dtype=np.float32)
    for r, c in track:
        ri, ci = round(r), round(c)
        if 0 <= ri < h and 0 <= ci < w:
            track_mask[ri, ci] = 1.0
    track_mask = track_mask[win[0][0]:win[0][1], win[1][0]:win[1][1]]

    # Structure mask
    struct_mask = np.zeros((h, w), dtype=np.float32)
    for r, c in allyx:
        ri, ci = round(r), round(c)
        if 0 <= ri < h and 0 <= ci < w:
            struct_mask[ri, ci] = 1.0
    struct_mask = struct_mask[win[0][0]:win[0][1], win[1][0]:win[1][1]]

    return tile.astype(np.float32), track_mask, struct_mask, win


def get_candidates_from_pmap(
    pmap: np.ndarray, threshold: float
) -> tuple[list[tuple[int, int]], float]:
    """Extract candidate coordinates from the vision module probability map.

    Finds the largest connected component in the binarized probability map,
    thins it, and returns its pixel positions.
    Corresponds to GetCandFromPmap in KymoButler.wl lines 164-171.

    Returns:
        (candidates, decision_prob) where candidates are (row, col) coords
        and decision_prob is the mean probability over the selected component.
    """
    from skimage.morphology import thin as skimage_thin

    binary = pmap > threshold
    structure = np.ones((3, 3), dtype=int)
    labeled, num = label(binary, structure=structure)
    if num == 0:
        return [], 0.0

    # Find largest component by area
    max_area = 0
    max_label = 0
    for i in range(1, num + 1):
        area = np.sum(labeled == i)
        if area > max_area:
            max_area = area
            max_label = i

    component_mask = labeled == max_label

    # Decision probability: mean pmap value over the component
    decision_prob = float(np.sum(component_mask * pmap) / np.sum(component_mask))

    # Thin the component and get coordinates
    thinned = skimage_thin(component_mask)
    coords = list(zip(*np.where(thinned)))

    return coords, decision_prob


def _sort_coords_greedy(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort coordinates into a connected chain using greedy nearest-neighbor.

    Tries both directions from the first point and returns the longer chain.
    Corresponds to SortCoords in KymoButler.wl lines 128-146.
    """
    if len(coords) <= 1:
        return coords

    coords_arr = np.array(coords, dtype=np.float64)

    def grow_chain(start_idx: int, prefer_min: bool) -> list[tuple[int, int]]:
        chain = [coords[start_idx]]
        remaining = set(range(len(coords))) - {start_idx}
        kdtree = KDTree(coords_arr)

        while remaining:
            last = np.array(chain[-1], dtype=np.float64)
            indices = kdtree.query_ball_point(last, r=1.5)
            indices = [i for i in indices if i in remaining]
            if not indices:
                break
            # Sort by last coordinate (col) and pick first or last
            candidates = [(i, coords[i]) for i in indices]
            if prefer_min:
                candidates.sort(key=lambda x: x[1][1])
            else:
                candidates.sort(key=lambda x: x[1][1], reverse=True)
            chosen_idx = candidates[0][0]
            chain.append(coords[chosen_idx])
            remaining.discard(chosen_idx)

        return chain

    chain_left = grow_chain(0, prefer_min=True)
    chain_right = grow_chain(0, prefer_min=False)

    return chain_left if len(chain_left) >= len(chain_right) else chain_right


def get_candidates(
    kym: np.ndarray,
    track: list[tuple[int, int]],
    allyx: np.ndarray,
    threshold: float,
    vision_net: torch.nn.Module,
    device: str = "cpu",
) -> tuple[list[tuple[int, int]], float]:
    """Use the vision module to find candidate next coordinates.

    Corresponds to GetCand in KymoButler.wl lines 183-243.

    Args:
        kym: Preprocessed kymograph (H, W).
        track: Current track coordinates.
        allyx: All skeleton coordinates.
        threshold: Vision module binarization threshold.
        vision_net: Loaded VisionNet model.
        device: Computation device.

    Returns:
        (new_coordinates, decision_prob) - new coords to extend track, and confidence.
    """
    dim = VISION_MODULE_TILE_SIZE
    pad_size = round(1 + dim / 2)

    # Pad the kymograph
    padkym = np.pad(kym, pad_size, mode="constant", constant_values=0.1)

    # Shift coordinates to padded space
    allyx_padded = allyx + pad_size
    track_padded = [(r + pad_size, c + pad_size) for r, c in track]

    last_padded = np.array(track_padded[-1], dtype=np.float64)

    # Find nearby skeleton pixels
    if len(allyx_padded) == 0:
        return [], 0.0
    kdtree = KDTree(allyx_padded)
    nearby_idx = kdtree.query_ball_point(last_padded, r=dim * 1.5)
    nearby = allyx_padded[nearby_idx]
    nearby = nearby[nearby[:, 0].argsort()]  # sort by row

    if len(nearby) == 0:
        return [], 0.0

    # Drop last point from track for tile extraction
    track_for_tile = track_padded[:-1]
    if len(track_for_tile) < 2:
        return [], 0.0

    # Get tile
    tile, track_mask, struct_mask, win = get_tile(padkym, track_for_tile, nearby, dim)

    # Ensure correct tile size
    if tile.shape[0] != dim or tile.shape[1] != dim:
        return [], 0.0

    # Run vision module
    tile_t = torch.from_numpy(tile).unsqueeze(0).unsqueeze(0).float().to(device)
    trk_t = torch.from_numpy(track_mask).unsqueeze(0).unsqueeze(0).float().to(device)
    struct_t = torch.from_numpy(struct_mask).unsqueeze(0).unsqueeze(0).float().to(device)

    vision_net.eval()
    with torch.no_grad():
        pmap_out = vision_net(tile_t, trk_t, struct_t)  # (1, 2, H, W)

    # Take foreground channel
    pmap = pmap_out[0, 1].cpu().numpy()  # (H, W) - second channel is foreground

    # Rescale coordinates to tile space
    win_offset = np.array([win[0][0], win[1][0]])
    last_in_tile = np.array(track_padded[-1]) - win_offset

    # Get candidates from probability map
    cands, decision_prob = get_candidates_from_pmap(pmap, threshold)
    if len(cands) == 0:
        return [], 0.0

    # Remove coordinates already in track
    track_set = set(map(tuple, track_padded))
    cands = [c for c in cands if tuple(np.array(c) + win_offset) not in track_set]

    if len(cands) < 3:
        return [], 0.0

    # Sort by distance to last track point
    cands.sort(key=lambda c: np.sqrt((c[0] - last_in_tile[0]) ** 2 + (c[1] - last_in_tile[1]) ** 2))

    # Check mean row is not going backwards
    cand_mean_row = np.mean([c[0] for c in cands])
    if cand_mean_row - last_in_tile[0] < -1:
        return [], 0.0

    # Sort into connected chain
    cands = _sort_coords_greedy(cands)

    # Bridge gap with shortest path if needed
    if len(cands) > 0:
        gap_dist = np.sqrt(
            (last_in_tile[0] - cands[0][0]) ** 2 + (last_in_tile[1] - cands[0][1]) ** 2
        )
        if gap_dist > 1.5:
            # Build binary image of nearby skeleton in tile space
            tile_struct = np.zeros((dim, dim), dtype=bool)
            for r, c in nearby:
                ri, ci = round(r - win_offset[0]), round(c - win_offset[1])
                if 0 <= ri < dim and 0 <= ci < dim:
                    tile_struct[ri, ci] = True
            start_rc = (round(last_in_tile[0]), round(last_in_tile[1]))
            end_rc = cands[0]
            if (0 <= start_rc[0] < dim and 0 <= start_rc[1] < dim
                    and 0 <= end_rc[0] < dim and 0 <= end_rc[1] < dim):
                bridge = find_shortest_path_on_skeleton(tile_struct, start_rc, end_rc)
                cands = bridge + cands

    # Limit to MAX_CANDIDATES
    cands = cands[:MAX_CANDIDATES]

    # Check mean row again after pathfinding
    if len(cands) > 0:
        cand_mean_row = np.mean([c[0] for c in cands])
        if cand_mean_row - last_in_tile[0] < -1:
            return [], 0.0

    # Convert back from tile space to original space (unpad)
    result = [(r + win_offset[0] - pad_size, c + win_offset[1] - pad_size) for r, c in cands]
    return result, decision_prob
