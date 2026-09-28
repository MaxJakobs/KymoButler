"""Exact port of Mathematica's ``Pruning[image, n]`` (integer ``n``, ``Padding -> 0``).

KymoButler's skeleton step in Mathematica is ``Pruning[Thinning@bin, n]``. Stripping every
endpoint ``n`` times (the previous port) is *not* what Mathematica does: it shortens every track
by ``n`` frames at both ends. This module follows Mathematica 14.3's own implementation,
``Image`MorphologicalOperationsDump`iPruning`` (method ``"nearest"``), read from the kernel::

    temp    = MorphologicalTransform[img, kernels, n]              (* 4 endpoint rules, in sequence, n times *)
    branchp = MorphologicalTransform[img, "SkeletonBranchPoints"]   (* branch points of the ORIGINAL image *)
    repeat until nothing changes:
        temp2   = temp3 - branchp * SkeletonEndPoints(res)
        endp    = MorphologicalTransform[temp2, endpRule]
        dilatep = Dilation[endp, 1]
        marker  = dilatep * branchp
        erased  = dilatep - GeodesicDilation[marker, dilatep]
        final   = erased + marker * dilatep
        temp3   = temp2 + final * img

Every ``MorphologicalTransform`` here is a 3x3 rule, given below by its Mathematica rule number:
bit ``k`` is the output for the 3x3 neighbourhood whose row-major bits (top-left = most
significant) spell ``k``. Verified pixel-exact against Mathematica on ~12,000 skeleton/n
combinations.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_propagation

# Rule numbers, from iPruning's source ("kernels" and the end-point marker rule) and from
# MorphologicalTransform[., "SkeletonEndPoints" | "SkeletonBranchPoints"] (both 3x3-local).
_KERNEL_RULES = (
    0xffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffee0000fffe0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffee0000ffff0000,
    0xffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ff230000,
    0xffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000fcff0000ffff0000fcfd0000,
    0xffff0000fffe0000ffff0000fffe0000ffff0000ffff0000ffff0000ffff0000ffff0000fffe0000ffff0000fffe0000ffff0000fffe0000ffff0000ffff0000,
)
_ENDP_RULE = 0x100000000000000010000000000000000000000110000000100000000000000010000000000000001000000000000030100000011000003df0000
_SKELETON_ENDPOINTS_RULE = 0x100000000000000010000000000000000000000110000000100000000000000010000000000000001000000000000030100000011000003de0000
_SKELETON_BRANCHPOINTS_RULE = 0x2e00007420000030220000f42000004cee0000fcfe000044000000fc20000022ae00003020000037ae0000f42000000cee0000002000000400000000000000

_WEIGHTS = (1 << np.arange(8, -1, -1)).reshape(3, 3)
_BOX = np.ones((3, 3), dtype=bool)


def _lut(rule: int) -> np.ndarray:
    return np.array([(rule >> k) & 1 for k in range(512)], dtype=bool)


_KERNELS = tuple(_lut(r) for r in _KERNEL_RULES)
_ENDP = _lut(_ENDP_RULE)
_SKEL_END = _lut(_SKELETON_ENDPOINTS_RULE)
_SKEL_BRANCH = _lut(_SKELETON_BRANCHPOINTS_RULE)


def morphological_transform(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """MorphologicalTransform[img, rule, Padding -> 0] for a 512-entry lookup table."""
    h, w = img.shape
    p = np.pad(img, 1).astype(np.int32)
    idx = np.zeros((h, w), dtype=np.int32)
    for i in range(3):
        for j in range(3):
            idx += p[i : i + h, j : j + w] * int(_WEIGHTS[i, j])
    return lut[idx]


def mma_pruning(skeleton: np.ndarray, n: int) -> np.ndarray:
    """Mathematica Pruning[skeleton, n]: remove short side branches, keep line ends."""
    img = np.asarray(skeleton, dtype=bool)
    if img.ndim != 2:
        raise ValueError(f"mma_pruning expects a 2-D binary image, got shape {img.shape}")
    n = int(n)
    if n <= 0 or not img.any():
        return img.copy()
    temp = img
    for _ in range(n):
        for k in _KERNELS:
            temp = morphological_transform(temp, k)
    branchp = morphological_transform(img, _SKEL_BRANCH)
    temp3, res = temp, None
    while res is None or not np.array_equal(temp3, res):
        res = temp3
        temp2 = temp3 & ~(branchp & morphological_transform(res, _SKEL_END))
        dilatep = binary_dilation(morphological_transform(temp2, _ENDP), structure=_BOX)
        marker = dilatep & branchp
        erased = dilatep & ~binary_propagation(marker, structure=_BOX, mask=dilatep)
        temp3 = temp2 | ((erased | (marker & dilatep)) & img)
    return res
