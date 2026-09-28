"""Graph-based shortest path on skeleton images.

Corresponds to FindShortPathImage in KymoButler.wl lines 113-126.
"""

from __future__ import annotations

from collections import deque

import numpy as np

_NEIGHBOURS_8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


def find_shortest_path_on_skeleton(
    binary: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    """Shortest path between two pixels of a binary skeleton, exactly as FindShortPathImage.

    Mathematica builds an *unweighted* graph whose vertices are the skeleton pixels plus
    ``start`` and ``end`` (labelled 1 and 2; every other foreground pixel gets the labels
    3, 4, ... in row-major order), with an edge between every pair of 8-adjacent vertices,
    and calls ``FindShortestPath[g, 1, 2]``. That is a breadth-first search: among the
    paths with the fewest hops it returns the one found by visiting each vertex's
    neighbours in ascending label order (validated 300/300 against Mathematica 14.3).

    Args:
        binary: Binary skeleton image (2D bool/int array).
        start: (row, col) start pixel (a vertex even if it is not on the skeleton).
        end: (row, col) end pixel (a vertex even if it is not on the skeleton).

    Returns:
        The (row, col) pixels from ``start`` to ``end`` inclusive, or [] if they are not
        connected.
    """
    h, w = binary.shape
    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))
    vertex = binary.astype(bool).copy()
    vertex[start] = True
    vertex[end] = True

    def label_key(p: tuple[int, int]) -> int:
        if p == start:
            return 0
        if p == end:
            return 1
        return 2 + p[0] * w + p[1]

    pred: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    queue = deque([start])
    while queue:
        v = queue.popleft()
        if v == end:
            break
        r, c = v
        nbrs = [
            (r + dr, c + dc)
            for dr, dc in _NEIGHBOURS_8
            if 0 <= r + dr < h and 0 <= c + dc < w and vertex[r + dr, c + dc]
        ]
        for u in sorted(nbrs, key=label_key):
            if u not in pred:
                pred[u] = v
                queue.append(u)

    if end not in pred:
        return []
    path = [end]
    while pred[path[-1]] is not None:
        path.append(pred[path[-1]])
    return path[::-1]
