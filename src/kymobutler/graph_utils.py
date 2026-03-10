"""Graph-based shortest path on skeleton images.

Corresponds to FindShortPathImage in KymoButler.wl lines 113-126.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path


def find_shortest_path_on_skeleton(
    binary: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    """Find shortest path between two points on a binary skeleton image.

    Algorithm (matching Mathematica's FindShortPathImage):
    1. Zero out start and end pixels in the binary image
    2. Label remaining connected components (each pixel gets a unique label)
    3. Assign start=1, end=2
    4. Build adjacency graph from neighboring labels
    5. Find shortest path from label 1 to label 2

    Args:
        binary: Binary skeleton image (2D bool/int array).
        start: (row, col) start pixel.
        end: (row, col) end pixel.

    Returns:
        List of (row, col) coordinates along the shortest path, or empty if no path exists.
    """
    bindat = binary.astype(np.int32).copy()
    h, w = bindat.shape

    # Zero out start and end
    bindat[start[0], start[1]] = 0
    bindat[end[0], end[1]] = 0

    # Renumber: each remaining foreground pixel gets a unique label starting at 3
    structure = np.ones((3, 3), dtype=int)
    labeled, _ = label(bindat, structure=structure)
    # Flatten labels to unique per-pixel IDs
    renumbered = np.zeros_like(bindat, dtype=np.int32)
    next_id = 3
    pixel_to_id = {}
    id_to_pixel = {}

    for r in range(h):
        for c in range(w):
            if labeled[r, c] > 0:
                renumbered[r, c] = next_id
                pixel_to_id[(r, c)] = next_id
                id_to_pixel[next_id] = (r, c)
                next_id += 1

    # Assign start=1, end=2
    renumbered[start[0], start[1]] = 1
    pixel_to_id[start] = 1
    id_to_pixel[1] = start
    renumbered[end[0], end[1]] = 2
    pixel_to_id[end] = 2
    id_to_pixel[2] = end

    n_vertices = next_id

    # Build adjacency graph (8-connectivity)
    graph = lil_matrix((n_vertices, n_vertices), dtype=np.float64)
    for r in range(h):
        for c in range(w):
            if renumbered[r, c] == 0:
                continue
            vid = renumbered[r, c]
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and renumbered[nr, nc] > 0:
                        nid = renumbered[nr, nc]
                        if vid != nid:
                            dist = 1.414 if (dr != 0 and dc != 0) else 1.0
                            graph[vid, nid] = dist
                            graph[nid, vid] = dist

    # Find shortest path from 1 to 2
    graph_csr = graph.tocsr()
    dist_matrix, predecessors = shortest_path(
        graph_csr, directed=False, indices=1, return_predecessors=True
    )

    if predecessors[2] < 0:
        return []

    # Reconstruct path
    path_ids = []
    current = 2
    while current != 1:
        path_ids.append(current)
        current = predecessors[current]
        if current < 0:
            return []
    path_ids.append(1)
    path_ids.reverse()

    return [id_to_pixel[pid] for pid in path_ids if pid in id_to_pixel]
