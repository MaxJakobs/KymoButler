"""Morphological processing for kymograph segmentation cleanup.

Corresponds to SmoothBin, SmoothBinUni, chewEnds, etc. in KymoButler.wl.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_hit_or_miss, label
from skimage.morphology import thin, remove_small_objects


def _hit_miss_union(binary: np.ndarray, kernels: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Apply multiple hit-miss transforms and return their union."""
    result = np.zeros_like(binary, dtype=bool)
    for structure1, structure2 in kernels:
        result |= binary_hit_or_miss(binary, structure1=structure1, structure2=structure2)
    return result


# --- Unidirectional smoothing kernels ---
# From KymoButler.wl line 42: SmoothBinUni
# Mathematica convention: 1=foreground, -1=background, 0=don't care
# scipy convention: structure1=foreground pattern, structure2=background pattern
_UNI_SMOOTH_KERNELS = [
    # {{0,1,0},{0,-1,1},{0,1,0}} -> center must be background, neighbors foreground
    (np.array([[0, 1, 0], [0, 0, 1], [0, 1, 0]]),
     np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])),
    # {{0,1,0},{1,-1,1},{0,0,0}}
    (np.array([[0, 1, 0], [1, 0, 1], [0, 0, 0]]),
     np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])),
    # {{0,1,0},{1,-1,0},{0,1,0}}
    (np.array([[0, 1, 0], [1, 0, 0], [0, 1, 0]]),
     np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])),
    # {{0,0,0},{1,-1,1},{0,1,0}}
    (np.array([[0, 0, 0], [1, 0, 1], [0, 1, 0]]),
     np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])),
]

# --- Bidirectional smoothing kernels ---
# From KymoButler.wl lines 293-389: SmoothBin
# Add kernels (gap filling): center=-1 (background), neighbors=1 (foreground)
_BI_ADD_KERNELS = [
    # {{0,1,1},{0,-1,1},{0,1,1}}
    (np.array([[0, 1, 1], [0, 0, 1], [0, 1, 1]]),
     np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])),
    # {{1,1,1},{1,-1,1},{0,0,0}}
    (np.array([[1, 1, 1], [1, 0, 1], [0, 0, 0]]),
     np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])),
    # {{1,1,0},{1,-1,0},{1,1,0}}
    (np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0]]),
     np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])),
    # {{0,0,0},{1,-1,1},{1,1,1}}
    (np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]]),
     np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])),
]

# Subtract kernels (isolated pixel removal): center=1 (foreground), marked neighbors=-1 (background)
_BI_SUB_KERNELS = [
    # {{0,-1,-1},{0,1,-1},{0,-1,-1}}
    (np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
     np.array([[0, 1, 1], [0, 0, 1], [0, 1, 1]])),
    # {{-1,-1,-1},{-1,1,-1},{0,0,0}}
    (np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
     np.array([[1, 1, 1], [1, 0, 1], [0, 0, 0]])),
    # {{-1,-1,0},{-1,1,0},{-1,-1,0}}
    (np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
     np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0]])),
    # {{0,0,0},{-1,1,-1},{-1,-1,-1}}
    (np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
     np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]])),
]

# Chew-end kernels (endpoint removal)
# {{-1,-1,-1},{-1,1,1},{-1,-1,-1}} -> right-facing endpoint
_CHEW_RIGHT = (
    np.array([[0, 0, 0], [0, 1, 1], [0, 0, 0]]),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]]),
)
# {{-1,-1,-1},{1,1,-1},{-1,-1,-1}} -> left-facing endpoint
_CHEW_LEFT = (
    np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]]),
    np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1]]),
)

# Seed detection kernel: top-facing endpoint
# {{-1,-1,-1},{-1,1,-1},{0,0,0}}
_SEED_KERNEL = (
    np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]),
    np.array([[1, 1, 1], [1, 0, 1], [0, 0, 0]]),
)


def smooth_binary_uni(binary: np.ndarray) -> np.ndarray:
    """Apply 4 hit-miss gap-filling transforms for unidirectional skeletons.

    Corresponds to SmoothBinUni in KymoButler.wl line 42.
    """
    return binary | _hit_miss_union(binary, _UNI_SMOOTH_KERNELS)


def smooth_binary_bi(binary: np.ndarray) -> np.ndarray:
    """Apply bidirectional smoothing: fill gaps and remove isolated pixels.

    Corresponds to SmoothBin in KymoButler.wl lines 293-389.
    """
    added = _hit_miss_union(binary, _BI_ADD_KERNELS)
    removed = _hit_miss_union(binary, _BI_SUB_KERNELS)
    return (binary | added) & ~removed


def chew_ends(binary: np.ndarray) -> np.ndarray:
    """Remove horizontal endpoints (pixels with only one horizontal neighbor).

    Corresponds to chewEnds in KymoButler.wl line 393.
    """
    right_ep = binary_hit_or_miss(binary, *_CHEW_RIGHT)
    left_ep = binary_hit_or_miss(binary, *_CHEW_LEFT)
    return binary & ~(right_ep | left_ep)


def chew_all_ends(binary: np.ndarray) -> np.ndarray:
    """Iteratively remove horizontal endpoints until stable.

    Corresponds to chewAllEnds in KymoButler.wl line 394.
    """
    old = binary
    new = chew_ends(old)
    while not np.array_equal(old, new):
        old = new
        new = chew_ends(old)
    return new


def detect_seeds(skeleton: np.ndarray) -> list[tuple[int, int]]:
    """Detect seed points (top-facing endpoints) on the skeleton.

    Seeds are pixels that have no foreground neighbors above them.
    Corresponds to HitMissTransform[chewAllEnds@paths, {{-1,-1,-1},{-1,1,-1},{0,0,0}}]
    in KymoButler.wl line 434.

    Returns:
        List of (row, col) coordinates sorted by row (top to bottom).
    """
    chewed = chew_all_ends(skeleton)
    seed_map = binary_hit_or_miss(chewed, *_SEED_KERNEL)
    coords = list(zip(*np.where(seed_map)))
    return sorted(coords, key=lambda c: c[0])


def _prune_branches(skeleton: np.ndarray, iterations: int) -> np.ndarray:
    """Remove branch endpoints iteratively (equivalent to Mathematica's Pruning).

    Each iteration removes all pixels that are endpoints (have only 1 neighbor
    in the 8-connected sense).
    """
    result = skeleton.copy()
    for _ in range(iterations):
        # Find endpoints: pixels with exactly 1 neighbor in 8-connectivity
        # Use a 3x3 structuring element
        from scipy.ndimage import convolve
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_count = convolve(result.astype(np.int32), kernel, mode="constant", cval=0)
        endpoints = result & (neighbor_count == 1)
        if not np.any(endpoints):
            break
        result = result & ~endpoints
    return result


def _filter_components(
    binary: np.ndarray, min_size: int, min_frames: int
) -> np.ndarray:
    """Filter connected components by pixel count and vertical span (frame span).

    Corresponds to SelectComponents[..., #Count>=minSz && BoundingBox span >= minFr]
    in KymoButler.wl.
    """
    # Use 8-connectivity (matching Mathematica's MorphologicalComponents default)
    structure = np.ones((3, 3), dtype=int)
    labeled, num_features = label(binary, structure=structure)
    result = np.zeros_like(binary, dtype=bool)
    for i in range(1, num_features + 1):
        component = labeled == i
        count = np.sum(component)
        rows = np.where(np.any(component, axis=1))[0]
        if len(rows) == 0:
            continue
        frame_span = rows[-1] - rows[0]
        if count >= min_size and frame_span >= min_frames:
            result |= component
    return result


def process_segmentation_uni(
    prediction: np.ndarray,
    original_dims: tuple[int, int],
    threshold: float = 0.2,
    min_size: int = 3,
    min_frames: int = 3,
) -> np.ndarray:
    """Full unidirectional morphological pipeline.

    Pipeline: binarize -> resize -> thin -> smooth(3x) -> thin -> prune(2) -> filter.
    Corresponds to UniKymoButlerTrack in KymoButler.wl line 61:
      Pruning[Thinning@SmoothBinUni^3@Thinning@ImageResize[Binarize[...],dim],2]
    """
    from skimage.transform import resize as sk_resize

    # Binarize and resize to original dimensions
    binary = prediction > threshold
    binary = sk_resize(binary.astype(np.float32), original_dims, order=0).astype(bool)

    # Thin first (skeletonize the blobs)
    binary = thin(binary)

    # Apply SmoothBinUni 3 times to fill gaps in skeleton
    for _ in range(3):
        binary = smooth_binary_uni(binary)

    # Thin again
    binary = thin(binary)

    # Prune short branches (2 iterations)
    binary = _prune_branches(binary, iterations=2)

    # Filter by size and frame span
    binary = _filter_components(binary, min_size, min_frames)

    return binary


def process_segmentation_bi(
    prediction: np.ndarray,
    original_dims: tuple[int, int],
    threshold: float = 0.2,
    min_size: int = 10,
    min_frames: int = 10,
) -> np.ndarray:
    """Full bidirectional morphological pipeline.

    Pipeline: binarize -> resize -> smooth(2x) -> thin -> prune(3) -> filter.
    Corresponds to BiKymoButlerTrack preprocessing in KymoButler.wl lines 428-431.
    """
    from skimage.transform import resize as sk_resize

    # Binarize and resize to original dimensions
    binary = prediction > threshold
    binary = sk_resize(binary.astype(np.float32), original_dims, order=0).astype(bool)

    # SmoothBin twice
    binary = smooth_binary_bi(smooth_binary_bi(binary))

    # Thin and prune
    binary = thin(binary)
    binary = _prune_branches(binary, iterations=3)

    # Filter components
    binary = _filter_components(binary, min_size, min_frames)

    return binary
