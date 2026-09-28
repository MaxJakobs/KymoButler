"""kymobutler.pruning.mma_pruning must reproduce Mathematica's Pruning[image, n] pixel for pixel.

Fixtures (tests/data/mma_pruning_cases.npz) are Mathematica 14.3 outputs on hand-made skeletons
(lines, spurs of 1-5 px, Y/cross junctions, loops, isolated segments) and on random thinned
blobs / random walks, for n in {1, 2, 3, 5, 8}.
"""

from pathlib import Path

import numpy as np
import pytest

from kymobutler.pruning import mma_pruning

_DATA = np.load(Path(__file__).parent / "data" / "mma_pruning_cases.npz")


def _cases():
    shapes, ns, off, bits = _DATA["shapes"], _DATA["n"], _DATA["offsets"], _DATA["bits"]
    for i, (shape, n) in enumerate(zip(shapes, ns)):
        size = int(np.prod(shape))
        b = bits[off[i] : off[i + 1]]
        half = len(b) // 2
        x = np.unpackbits(b[:half])[:size].reshape(shape).astype(bool)
        y = np.unpackbits(b[half:])[:size].reshape(shape).astype(bool)
        yield x, int(n), y


def test_matches_mathematica_on_all_fixture_cases():
    bad = [(i, n) for i, (x, n, want) in enumerate(_cases()) if not np.array_equal(mma_pruning(x, n), want)]
    assert not bad, f"{len(bad)} cases differ from Mathematica, first: {bad[:5]}"


def test_keeps_plain_lines_and_line_ends():
    img = np.zeros((30, 30), bool)
    img[3:28, 10] = True
    assert np.array_equal(mma_pruning(img, 3), img)  # the old port removed 3 px at each end


def test_removes_short_spur():
    img = np.zeros((30, 30), bool)
    img[3:28, 10] = True
    line = img.copy()
    img[15, 11:13] = True  # 2 px spur
    assert np.array_equal(mma_pruning(img, 3), line)


def test_trivial_inputs():
    assert not mma_pruning(np.zeros((5, 5), bool), 3).any()
    img = np.eye(6, dtype=bool)
    assert np.array_equal(mma_pruning(img, 0), img)
    with pytest.raises(ValueError):
        mma_pruning(np.zeros((2, 2, 2), bool), 3)
