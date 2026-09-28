"""Regression tests against Mathematica BiKymoButler references (public test images only).

The references in tests/data/reference/ were produced by Mathematica 14.3 with
legacy/packages/KymoButler.wl and the standard networks (see manifest.json). Each test feeds
the Python stage exactly Mathematica's input for that stage and requires exactly
Mathematica's output; the last test runs the whole production pipeline from the image file.
Pruning itself is covered by tests/test_pruning.py.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from skimage.morphology import thin

import kymobutler.morphology as morph
from kymobutler.morphology import (
    _filter_components,
    process_segmentation_bi,
    resize_binary,
    smooth_binary_bi,
)
from kymobutler.tracking import track_paths_bidirectional

REF = Path(__file__).parent / "data" / "reference"
MANIFEST = json.loads((REF / "manifest.json").read_text())
CASES = MANIFEST["cases"]
_MODEL_DIRS = (Path(__file__).parent.parent / "models", Path.home() / ".kymobutler" / "models")
_DECNET = next((d / "decision_module.onnx" for d in _MODEL_DIRS if (d / "decision_module.onnx").exists()), None)
_BINET = next((d / "bidirectional_seg.onnx" for d in _MODEL_DIRS if (d / "bidirectional_seg.onnx").exists()), None)


def _case_id(case):
    return f"{Path(case['image']).stem}_t{case['threshold']}_s{case['min_size']}_f{case['min_frames']}"


def _arrays(case):
    return np.load(REF / case["arrays"])


def _ptag(case):
    return f"t{case['threshold']}_s{case['min_size']}_f{case['min_frames']}"


def test_resize_binary_matches_mathematica():
    data = np.load(REF / "resize_mma.npz")
    inputs = np.unpackbits(data["inputs"]).astype(bool)
    outputs = np.unpackbits(data["outputs"]).astype(bool)
    i = o = 0
    for hi, wi, ho, wo in data["shapes"]:
        b = inputs[i:i + hi * wi].reshape(hi, wi)
        want = outputs[o:o + ho * wo].reshape(ho, wo)
        i, o = i + hi * wi, o + ho * wo
        assert np.array_equal(resize_binary(b, (ho, wo)), want), (hi, wi, ho, wo)
    assert len(data["shapes"]) == 424


def test_resize_binary_identity_and_unsupported():
    b = np.random.default_rng(0).random((20, 30)) > 0.5
    assert np.array_equal(resize_binary(b, (20, 30)), b)
    with pytest.raises(NotImplementedError):
        resize_binary(b, (10, 30))  # downsampling by 2x
    with pytest.raises(NotImplementedError):
        resize_binary(np.ones((51, 3), bool), (50, 3))  # a centre on a pixel boundary


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_binarize_resize_smooth_matches_mathematica(case):
    a = _arrays(case)
    thr = f"t{case['threshold']}"
    smoothed = smooth_binary_bi(
        smooth_binary_bi(resize_binary(a["pred"] > case["threshold"], a["kympreproc"].shape))
    )
    assert np.array_equal(smoothed, a[f"smoothed_{thr}"])
    assert np.array_equal(thin(a[f"smoothed_{thr}"]), a[f"thinned_{thr}"])


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_component_filter_matches_mathematica(case):
    a = _arrays(case)
    paths = _filter_components(a[f"pruned_t{case['threshold']}"], case["min_size"], case["min_frames"])
    assert np.array_equal(paths, a[f"paths_{_ptag(case)}"])


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_skeleton_pipeline_with_mathematica_pruning(case, monkeypatch):
    """process_segmentation_bi end to end, with Mathematica's Pruning result plugged in."""
    a = _arrays(case)
    thr = f"t{case['threshold']}"
    pruned = a[f"pruned_{thr}"]

    def mma_pruning(skeleton, n):
        assert n == 3
        assert np.array_equal(skeleton, a[f"thinned_{thr}"])
        return pruned

    monkeypatch.setattr(morph, "prune_skeleton", mma_pruning)
    paths = process_segmentation_bi(
        a["pred"], a["kympreproc"].shape, case["threshold"], case["min_size"], case["min_frames"]
    )
    assert np.array_equal(paths, a[f"paths_{_ptag(case)}"])


@pytest.mark.skipif(_DECNET is None, reason="decision_module.onnx not found")
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_tracks_on_mathematica_skeleton(case):
    """BiKymoButlerTrack from Mathematica's skeleton and kympreproc: identical tracks, in order."""
    from kymobutler.models.weights import _load_onnx_model

    a = _arrays(case)
    decnet = _load_onnx_model("decnet", _DECNET, "cpu")
    tracks = track_paths_bidirectional(
        a[f"paths_{_ptag(case)}"], a["kympreproc"], 0.5, decnet, case["min_size"], case["min_frames"]
    )
    got = [[[r + 1, c + 1] for r, c in t.points] for t in tracks]
    want = json.loads((REF / case["tracks"]).read_text())
    assert got == want


@pytest.mark.skipif(_DECNET is None or _BINET is None, reason="bidirectional_seg.onnx / decision_module.onnx not found")
@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_bidirectional_end_to_end_from_image(case):
    """BiKymoButler from the image file (preprocessing, net, skeleton, exact Pruning, tracking):
    identical tracks, in Mathematica's order."""
    from kymobutler.models.weights import _load_onnx_model
    from kymobutler.segmentation import segment_bidirectional
    from kymobutler.tracking import track_bidirectional

    binet = _load_onnx_model("binet", _BINET, "cpu")
    decnet = _load_onnx_model("decnet", _DECNET, "cpu")
    image = Path(__file__).parent.parent / case["image"]
    was_neg, _, pre, pred = segment_bidirectional(str(image), binet, "cpu")
    tracks = track_bidirectional(
        pred, pre, was_neg, case["threshold"], 0.5, decnet, case["min_size"], case["min_frames"], "cpu"
    )
    got = [[[r + 1, c + 1] for r, c in t.points] for t in tracks]
    assert got == json.loads((REF / case["tracks"]).read_text())
