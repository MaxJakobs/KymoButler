"""Morphological processing between the segmentation networks and tracking.

Ports of KymoButler.wl's SmoothBin, SmoothBinUni, chewEnds / chewAllEnds, the seed
HitMissTransform, and of the Mathematica built-ins the pipelines use on binary images:
``ImageResize`` of a binary image (``resize_binary``), ``Pruning`` (``prune_skeleton``, exact
port in :mod:`kymobutler.pruning`, used by both pipelines), ``SelectComponents`` and
``MorphologicalComponents`` with Mathematica's label numbering (``morphological_components``).
Hit-miss transforms use ``Padding -> 0`` like the Wolfram code: pixels outside the image are
background.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_hit_or_miss, find_objects, label
from skimage.morphology import thin

from kymobutler.pruning import mma_pruning


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

    Corresponds to SmoothBinUni in KymoButler.wl line 42 (``Padding -> 0``): a background
    pixel becomes foreground when all three foreground cells of one of the kernels are set.
    Computed with shifted views (equal to the union of the four hit-miss transforms).
    """
    b = np.asarray(binary, dtype=bool)
    h, w = b.shape
    p = np.pad(b, 1)
    added = np.zeros_like(b)
    for fg_cells, _ in _UNI_SMOOTH_KERNELS:
        hit = ~b
        for dr, dc in zip(*np.nonzero(fg_cells)):
            hit &= p[dr : dr + h, dc : dc + w]
        added |= hit
    return b | added


def smooth_binary_bi(binary: np.ndarray) -> np.ndarray:
    """SmoothBin (KymoButler.wl lines 293-389, ``Padding -> 0``).

    ``out + HitMissTransform[out, add] - HitMissTransform[out, sub]``, both transforms of the
    input: a background pixel closed in by foreground on three sides (the ``add`` kernels) is
    set, a foreground pixel with background on three sides (the ``sub`` kernels) is cleared.
    """
    added = _hit_miss_union(binary, _BI_ADD_KERNELS)
    removed = _hit_miss_union(binary, _BI_SUB_KERNELS)
    return (binary | added) & ~removed


def chew_ends(binary: np.ndarray) -> np.ndarray:
    """chewEnds (KymoButler.wl line 393, ``Padding -> 0``): remove every pixel whose only
    8-neighbour is its left or its right neighbour (the end of a horizontal run)."""
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
    """Seed points: ``HitMissTransform[chewAllEnds@paths, {{-1,-1,-1},{-1,1,-1},{0,0,0}},
    Padding -> 0]`` (KymoButler.wl lines 399, 434), the pixels of ``chewAllEnds@paths`` with
    no foreground above them or beside them (only below, if anywhere).

    Returns:
        (row, col) coordinates sorted by row, then column: ``SortBy[seeds, First]`` breaks
        ties in canonical order.
    """
    chewed = chew_all_ends(skeleton)
    seed_map = binary_hit_or_miss(chewed, *_SEED_KERNEL)
    coords = list(zip(*np.where(seed_map)))
    return sorted(coords, key=lambda c: c[0])


def prune_skeleton(skeleton: np.ndarray, n: int) -> np.ndarray:
    """Mathematica ``Pruning[skeleton, n]``, used by both pipelines: BiKymoButlerTrack with
    n=3 (KymoButler.wl line 431), UniKymoButlerTrack with n=2 (lines 61-62). Exact port, see
    :mod:`kymobutler.pruning`."""
    return mma_pruning(skeleton, n)


def _select_components(binary: np.ndarray, min_count: int, min_rows: int) -> np.ndarray:
    """Keep the 8-connected components with ``count >= min_count`` pixels whose bounding box
    is at least ``min_rows`` rows tall (``rows[-1] - rows[0] + 1``)."""
    labeled, num = label(binary, structure=np.ones((3, 3), dtype=int))
    if num == 0:
        return np.zeros(binary.shape, dtype=bool)
    counts = np.bincount(labeled.ravel(), minlength=num + 1)
    heights = np.zeros(num + 1, dtype=np.int64)
    for i, sl in enumerate(find_objects(labeled), start=1):
        if sl is not None:
            heights[i] = sl[0].stop - sl[0].start
    keep = (counts >= min_count) & (heights >= min_rows)
    keep[0] = False
    return keep[labeled]


def _filter_components(
    binary: np.ndarray, min_size: int, min_frames: int
) -> np.ndarray:
    """Filter connected components by pixel count and frame span.

    Mathematica: ``SelectComponents[img, #Count >= minSz &&
    #BoundingBox[[2,2]] - #BoundingBox[[1,2]] >= minFr &]`` (8-connectivity). The
    BoundingBox is in pixel-edge coordinates, so its height is the number of rows the
    component spans (``rows[-1] - rows[0] + 1``), not ``rows[-1] - rows[0]``.
    """
    return _select_components(binary, min_size, min_frames)


MIN_IMAGE_SIDE = 9


def check_image_size(shape: tuple[int, ...]) -> None:
    """Raise the one error KymoButler gives for kymographs that are too small to analyse.

    Both pipelines run the network at ``16*Round[d/16]`` pixels per side (Round is half to
    even), which is 0 for a side of 8 px or less: Mathematica 14.3 cannot segment such images
    either (``ImageResize::imgrssz``, then ``NetGraph::invindata2``); 9 px (network side 16)
    works. Checked by ``segment_bidirectional`` / ``segment_unidirectional`` before the network
    runs, and by ``process_segmentation_bi`` / ``process_segmentation_uni`` before resizing a
    network-resolution prediction, so the two pipelines fail alike and up front.

    Args:
        shape: (height, width) of the kymograph.

    Raises:
        ValueError: if a side is shorter than 9 px.
    """
    h, w = int(shape[0]), int(shape[1])
    if h < MIN_IMAGE_SIDE or w < MIN_IMAGE_SIDE:
        raise ValueError(
            f"KymoButler needs an image of at least {MIN_IMAGE_SIDE} x {MIN_IMAGE_SIDE} pixels; "
            f"this image is {w} x {h}"
        )


def resize_binary(binary: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    """Mathematica ``ImageResize[binaryImage, {w, h}]`` (default resampling) for a binary image.

    Separable OR resampling: along each axis, output pixel ``j`` covers the input interval
    ``[j*s, (j+1)*s)`` with ``s = n_in/n_out`` and is the OR of every input pixel whose centre
    ``k + 1/2`` lies in it. If no centre does (upsampling), it takes the input pixel nearest to
    the output centre ``(j + 1/2)*s - 1/2``, or the OR of both pixels on an exact tie. Identity
    when the shapes already agree. Exact integer arithmetic throughout.

    Identical to Mathematica 14.3 on every net-resolution -> image resize ``16*Round[d/16] ->
    d`` (the only resizes UniKymoButlerTrack and BiKymoButlerTrack do) that was checked: every
    d = 9..2000 on each axis, 160 sizes from 2001 to 20000 and 156 random 2D cases up to 4000 x
    4000 px, 4300/4300 random binary images. Outside that regime this rule is not always
    Mathematica's, so it raises NotImplementedError instead of guessing when downsampling by 2x
    or more (Mathematica differs at 3x) or when an input centre falls exactly on an output pixel
    boundary (Mathematica's tie-breaking there varies). Neither happens for net-resolution ->
    image resizes of images at least 9 px on each side; smaller kymographs are rejected before
    (``check_image_size``).
    """
    binary = np.asarray(binary, dtype=bool)
    out_shape = (int(out_shape[0]), int(out_shape[1]))
    if binary.shape == out_shape:
        return binary.copy()
    if min(out_shape) < 1 or min(binary.shape) < 1:
        raise ValueError(f"cannot resize a {binary.shape} image to {out_shape}")
    for n_in, n_out in zip(binary.shape, out_shape):
        j = np.arange(1, n_out, dtype=np.int64)
        if n_in >= 2 * n_out or np.any((2 * j * n_in - n_out) % (2 * n_out) == 0):
            raise NotImplementedError(
                f"resizing {binary.shape} -> {out_shape}: Mathematica's binary ImageResize is only "
                "reproduced for net-resolution -> image resizes (16*Round[d/16] -> d, d >= 9)"
            )
    out = binary
    for axis in (0, 1):
        lo, hi = _resize_ranges(binary.shape[axis], out_shape[axis])
        # below 2x downsampling every output pixel ORs one or two adjacent input pixels
        if not np.all((hi - lo >= 0) & (hi - lo <= 1)):
            raise RuntimeError(f"resize_binary invariant violated for {binary.shape} -> {out_shape}")
        out = np.take(out, lo, axis=axis) | np.take(out, hi, axis=axis)
    return out


def _resize_ranges(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """Inclusive input index range [lo, hi] OR-ed into each output pixel (see resize_binary)."""
    j = np.arange(n_out, dtype=np.int64)
    den = 2 * n_out
    # centres k + 1/2 in [j*s, (j+1)*s)  <=>  x_j <= k < x_{j+1},  x_j = (2*j*n_in - n_out)/den
    lo = -((n_out - 2 * j * n_in) // den)  # ceil(x_j)
    hi = -((n_out - 2 * (j + 1) * n_in) // den) - 1  # ceil(x_{j+1}) - 1
    empty = lo > hi
    # nearest input centre to the output centre c = ((2j+1)*n_in - n_out)/den; both on a tie
    num = (2 * j + 1) * n_in - n_out
    f = num // den
    tie = 2 * (num - f * den) == den
    nearest = (2 * num + den) // (2 * den)  # floor(c + 1/2)
    lo = np.where(empty, np.where(tie, f, nearest), lo)
    hi = np.where(empty, np.where(tie, f + 1, nearest), hi)
    return np.clip(lo, 0, n_in - 1), np.clip(hi, 0, n_in - 1)


def _select_components_uni(binary: np.ndarray, min_size: float, min_frames: int) -> np.ndarray:
    """``SelectComponents[img, #Length >= minSz && #BoundingBox[[2,2]] - #BoundingBox[[1,2]] >=
    minFr &]`` (8-connectivity), as in UniKymoButlerTrack (KymoButler.wl lines 61-62).

    ComponentMeasurements' ``"Length"`` is ``max(4 * sqrt(lambda_max), 0.5)`` with ``lambda_max``
    the largest eigenvalue of the population covariance of the component's pixel centres
    (verified on 1287 real skeleton components and 13 synthetic shapes). The BoundingBox height
    is the number of rows the component spans.
    """
    labeled, num = label(binary, structure=np.ones((3, 3), dtype=int))
    if num == 0:
        return np.zeros(binary.shape, dtype=bool)
    rr, cc = np.nonzero(labeled)
    lab = labeled[rr, cc]
    cnt = np.bincount(lab, minlength=num + 1).astype(np.float64)
    cnt[0] = 1.0  # background, never kept

    def per_label(values: np.ndarray) -> np.ndarray:
        return np.bincount(lab, weights=values, minlength=num + 1) / cnt

    # centred second moments (two-pass: exact for the integer-valued lines and pairs whose
    # Length is an integer, e.g. 2 px -> 2, 7 px -> 8)
    dr = rr - per_label(rr.astype(np.float64))[lab]
    dc = cc - per_label(cc.astype(np.float64))[lab]
    vrr, vcc, vrc = per_label(dr * dr), per_label(dc * dc), per_label(dr * dc)
    lmax = (vrr + vcc) / 2 + np.sqrt(((vrr - vcc) / 2) ** 2 + vrc**2)
    length = np.maximum(4 * np.sqrt(np.maximum(lmax, 0.0)), 0.5)
    heights = np.zeros(num + 1, dtype=np.int64)
    for i, sl in enumerate(find_objects(labeled), start=1):
        if sl is not None:
            heights[i] = sl[0].stop - sl[0].start
    keep = (length >= min_size) & (heights >= min_frames)
    keep[0] = False
    return keep[labeled]


def morphological_components(binary: np.ndarray) -> tuple[np.ndarray, int]:
    """``MorphologicalComponents[binary]`` (8-connectivity), numbered exactly like Mathematica.

    The partition is that of ``scipy.ndimage.label``; the numbering is not: Mathematica does
    not number the components by their first pixel (that order agrees with Mathematica's on
    only 59 % of random binary images and 26 % of thinned noise). Mathematica 14.3 labels by a
    raster scan (top row first, left to right) with a union-find over provisional labels:

    - a foreground pixel takes the provisional label of its first labelled neighbour in the
      order W, N, NE, NW, or a new provisional label (numbered in scan order) if it has none;
    - every further labelled neighbour unites the two label sets, union by size (size = the
      number of provisional labels in the set): the root of the larger set becomes the root;
      on a tie the current pixel's set goes under the neighbour's;
    - the components are numbered by increasing provisional label of their final roots.

    Only a pixel whose N neighbour is background, NE foreground and W or NW foreground can
    join two different sets (N is 8-adjacent to W, NW and NE, so with N set they already share
    its set; W and NW are adjacent; NE is adjacent to neither): the union of the W (or, with W
    background, NE) set with the NE (or NW) set. Only those events are replayed in Python.

    Identical label images to Mathematica 14.3's, for "Bit" and "Real32" images alike, on six
    independently generated sets, 135387/135387 images and 1.7 Gpx in all (the last one, 11822
    images and 422 Mpx, generated after this implementation was fixed): every binary pattern of
    4 x 4 px or smaller and of 2 x 9, 9 x 2, 3 x 6, 6 x 3, 3 x 7, 4 x 5 and 5 x 4 px; random
    images of every size (1 px to 25 Mpx) and density, 1- to 3-pixel strips, percolation
    clusters, thinned and skeletonized blobs and noise, random walks, rings, combs, trees,
    diagonal segments, V cascades, checkerboards, skeletons through the uni and bi
    morphological chains, CatchStraddlers' ``Dilation[pathstmp, 3]``, and skeletons, pathstmp
    and straddler dilations of real runs on the public test images; plus 48/48 unidirectional
    skeletons of real kymographs. ``ComponentMeasurements`` lists the components of a binary
    image in this order (GetCandFromPmap's first largest component: 2043/2043 decision-module
    maps, 413 with tied largest components) and the masks of a label matrix in label order
    (CatchStraddlers' seed lists: 2252/2252 identical).

    Returns:
        (labels, n): int64 label image (0 = background, 1..n) and the number of components.
    """
    b = np.asarray(binary, dtype=bool)
    h, w = b.shape
    labels = np.zeros((h, w), dtype=np.int64)
    fg = np.flatnonzero(b)
    n_fg = fg.size
    if n_fg == 0:
        return labels, 0
    padded = np.pad(b, 1)
    r, c = np.divmod(fg, w)
    offsets = ((0, -1), (-1, 0), (-1, 1), (-1, -1))  # W, N, NE, NW: all scanned before p
    nb = np.stack([padded[r + 1 + dr, c + 1 + dc] for dr, dc in offsets], axis=1)
    rank = np.full(h * w, -1, dtype=np.int64)
    rank[fg] = np.arange(n_fg)
    nb_rank = np.stack(
        [np.where(nb[:, i], rank[np.clip(fg + dr * w + dc, 0, h * w - 1)], -1)
         for i, (dr, dc) in enumerate(offsets)],
        axis=1,
    )
    has = nb.any(axis=1)
    first = np.argmax(nb, axis=1)
    # provisional label: a new one for pixels without labelled neighbours (raster order),
    # otherwise that of the first labelled neighbour, i.e. of the start of the pointer chain
    ptr = np.where(has, nb_rank[np.arange(n_fg), first], np.arange(n_fg))
    while True:
        nxt = ptr[ptr]
        if np.array_equal(nxt, ptr):
            break
        ptr = nxt
    start_id = np.cumsum(~has)  # provisional id (1-based) of each start pixel
    prov = start_id[ptr]
    n_prov = int(start_id[-1])

    # merge events (see above), in scan order: (current set, neighbour set); on a size tie the
    # neighbour's root wins. W set: NE is the neighbour; W background: NE's set meets NW.
    w_, n_, ne, nw = nb.T
    ev = np.flatnonzero(~n_ & ne & (w_ | nw))
    cur = np.where(w_[ev], nb_rank[ev, 0], nb_rank[ev, 2])
    oth = np.where(w_[ev], nb_rank[ev, 2], nb_rank[ev, 3])
    a, o = prov[cur], prov[oth]
    keep = a != o
    a, o = a[keep], o[keep]
    if a.size:
        # a later union of two labels already united is a no-op: keep first occurrences only
        pair = np.minimum(a, o) * (n_prov + 1) + np.maximum(a, o)
        _, idx = np.unique(pair, return_index=True)
        idx.sort()
        a, o = a[idx], o[idx]
    parent = list(range(n_prov + 1))
    size = [1] * (n_prov + 1)
    for x, y in zip(a.tolist(), o.tolist()):
        while parent[x] != x:  # find with path halving (roots and sizes are unaffected)
            parent[x] = parent[parent[x]]
            x = parent[x]
        while parent[y] != y:
            parent[y] = parent[parent[y]]
            y = parent[y]
        if x == y:
            continue
        if size[x] > size[y]:
            parent[y] = x
            size[x] += size[y]
        else:
            parent[x] = y
            size[y] += size[x]
    roots = np.asarray(parent, dtype=np.int64)
    while True:
        nxt = roots[roots]
        if np.array_equal(nxt, roots):
            break
        roots = nxt
    uniq, comp = np.unique(roots[prov], return_inverse=True)
    labels.ravel()[fg] = comp.ravel() + 1
    return labels, int(uniq.size)


def process_segmentation_uni(
    prediction: np.ndarray,
    original_dims: tuple[int, int],
    threshold: float = 0.2,
    min_size: int = 3,
    min_frames: int = 3,
) -> np.ndarray:
    """Full unidirectional morphological pipeline (UniKymoButlerTrack, KymoButler.wl 61-62)::

        SelectComponents[Pruning[Thinning@SmoothBinUni@SmoothBinUni@SmoothBinUni@Thinning@
            ImageResize[Binarize[out["ant"], binthresh], dim], 2],
            #Length >= minSz && #BoundingBox[[2,2]] - #BoundingBox[[1,2]] >= minFr &]

    Args:
        prediction: 'ant' or 'ret' probability map at the network's resolution (as returned by
            ``segment_unidirectional``), or already at ``original_dims``.
        original_dims: (height, width) of the kymograph.

    Raises:
        ValueError: for a network-resolution prediction of a kymograph smaller than 9 x 9 px,
            which Mathematica cannot segment (``check_image_size``).
    """
    if np.shape(prediction) != tuple(int(d) for d in original_dims):
        check_image_size(original_dims)
    # Binarize (strictly above the threshold, compared in double precision like Mathematica)
    # at network resolution, then resize the binary image exactly like ImageResize.
    binary = resize_binary(np.asarray(prediction, dtype=np.float64) > threshold, original_dims)

    # Thin first (skeletonize the blobs)
    binary = thin(binary)

    # Apply SmoothBinUni 3 times to fill gaps in skeleton
    for _ in range(3):
        binary = smooth_binary_uni(binary)

    # Thin again, then Pruning[..., 2] (it does remove short spurs of these skeletons)
    binary = thin(binary)
    binary = prune_skeleton(binary, 2)

    # Filter by Length and frame span
    return _select_components_uni(binary, min_size, min_frames)


def process_segmentation_bi(
    prediction: np.ndarray,
    original_dims: tuple[int, int],
    threshold: float = 0.2,
    min_size: int = 10,
    min_frames: int = 10,
) -> np.ndarray:
    """Full bidirectional morphological pipeline (BiKymoButlerTrack, KymoButler.wl 428-431):

        out = ImageResize[Binarize[pred, binthresh], dim]
        out = SmoothBin@SmoothBin@out
        paths = SelectComponents[Pruning[Thinning@out, 3], #Count >= minSz && span >= minFr &]

    Args:
        prediction: Foreground probability map at the network's resolution (as returned by
            ``segment_bidirectional``), or already at ``original_dims``.
        original_dims: (height, width) of the kymograph.

    Raises:
        ValueError: for a network-resolution prediction of a kymograph smaller than 9 x 9 px,
            which Mathematica cannot segment (``check_image_size``).
    """
    if np.shape(prediction) != tuple(int(d) for d in original_dims):
        check_image_size(original_dims)
    # Binarize (strictly above the threshold, compared in double precision like Mathematica)
    # at network resolution, then resize the binary image exactly like ImageResize.
    binary = resize_binary(np.asarray(prediction, dtype=np.float64) > threshold, original_dims)

    # SmoothBin twice
    binary = smooth_binary_bi(smooth_binary_bi(binary))

    # Thin and prune
    binary = thin(binary)
    binary = prune_skeleton(binary, 3)

    # Filter components
    binary = _filter_components(binary, min_size, min_frames)

    return binary
