"""Regression tests against Mathematica UniKymoButler and the legacy uni/bi classifier (public
test images only).

The references in tests/data/reference/uni_mma.{npz,json} were produced by Mathematica 14.3 with
legacy/packages/KymoButler.wl and the standard networks on unitest.png, unitest2.png and
bitest.png (see the "generator" field of uni_mma.json). Stage tests feed each Python stage
exactly Mathematica's input for that stage and require exactly Mathematica's output; the
end-to-end tests start from the image file.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from skimage.morphology import thin

import kymobutler.morphology as morph
from kymobutler.classification import (
    classifier_input,
    classifier_input_from_pixels,
    classify_direction,
    classify_probabilities,
)
from kymobutler.morphology import (
    _select_components_uni,
    morphological_components,
    process_segmentation_uni,
    resize_binary,
    smooth_binary_uni,
)
from kymobutler.preprocessing import (
    load_and_preprocess,
    load_and_preprocess_mma,
    load_grayscale,
    read_image,
    resize_to_multiple_of_16,
)
from kymobutler.pruning import mma_pruning
from kymobutler.tracking import _uni_tracks

DATA = Path(__file__).parent / "data"
REF = DATA / "reference"
META = json.loads((REF / "uni_mma.json").read_text())
ARR = np.load(REF / "uni_mma.npz")
IMAGES = sorted(META["shapes"])
STAGE_CASES = [(img, k, thr) for img in IMAGES for k in ("ant", "ret") for thr in ("t20", "t15")]
SELECTIONS = ((3, 3), (10, 10), (10, 3), (3, 10))
_MODEL_DIRS = (Path(__file__).parent.parent / "models", Path.home() / ".kymobutler" / "models")


def _model(name):
    return next((d / name for d in _MODEL_DIRS if (d / name).exists()), None)


def _bits(key, shape):
    return np.unpackbits(ARR[key])[: shape[0] * shape[1]].reshape(shape).astype(bool)


def _dims(img):
    return tuple(META["shapes"][img]["dims"])


def _stage(img, k, thr, name):
    shape = tuple(META["shapes"][img]["net"]) if name == "b0" else _dims(img)
    return _bits(f"{img}_{k}_{thr}_{name}", shape)


def _case_id(case):
    return "_".join(case)


def _mma_tracks(img, ptag, k):
    return [[(r - 1, c - 1) for r, c in t] for t in META["tracks"][img][ptag][k]]


def _parse(ptag):
    t, s, f = ptag.split("_")
    return int(t[1:]) / 100, int(s[1:]), int(f[1:])


# ------------------------------------------------------------------------------ preprocessing
@pytest.mark.parametrize("img", ["unitest2", "bitest"])
def test_rawkym_matches_mathematica(img):
    raw, levels = load_grayscale(DATA / f"{img}.png")
    assert levels == 255
    assert np.array_equal(np.rint(raw * 255).astype(np.uint8), ARR[f"{img}_rawkym"])


@pytest.mark.parametrize("img", ["unitest2", "bitest"])
def test_negkym_matches_mathematica(img):
    """negkym is the bidirectional kympreproc too (same code in BiKymoButlerSegment). It is a
    Real32 image in Mathematica: identical float32 values."""
    want = np.load(REF / f"{img}_mma.npz")["kympreproc"]
    pre, _, _ = load_and_preprocess(DATA / f"{img}.png")
    assert want.dtype == np.float32 and np.array_equal(pre, want)


def test_net_input_resize_matches_mathematica():
    pre, _, _, _ = load_and_preprocess_mma(DATA / "unitest2.png")
    netin = resize_to_multiple_of_16(pre)
    assert netin.shape == tuple(META["shapes"]["unitest2"]["net"])
    assert np.array_equal(netin, ARR["unitest2_netin"])  # float32, bit for bit


# ------------------------------------------------------------------------------ skeleton stages
@pytest.mark.parametrize("case", STAGE_CASES, ids=_case_id)
def test_uni_skeleton_stages_match_mathematica(case):
    img, k, thr = case
    g = lambda name: _stage(img, k, thr, name)  # noqa: E731
    assert np.array_equal(resize_binary(g("b0"), _dims(img)), g("r"))
    assert np.array_equal(thin(g("r")), g("t1"))
    s = g("t1")
    for _ in range(3):
        s = smooth_binary_uni(s)
    assert np.array_equal(s, g("s3"))
    assert np.array_equal(thin(g("s3")), g("t2"))
    # Pruning[., 2] removes pixels here (the exact port, kymobutler.pruning)
    assert np.array_equal(mma_pruning(g("t2"), 2), g("p"))


@pytest.mark.parametrize("case", STAGE_CASES, ids=_case_id)
def test_uni_select_components_matches_mathematica(case):
    img, k, thr = case
    p = _stage(img, k, thr, "p")
    for ms, mf in SELECTIONS:
        sel = _stage(img, k, thr, f"s{ms}_f{mf}_selL")
        assert np.array_equal(_select_components_uni(p, ms, mf), sel), (ms, mf)


@pytest.mark.parametrize("case", STAGE_CASES, ids=_case_id)
def test_morphological_components_order_matches_mathematica(case):
    img, k, thr = case
    labels, n = morphological_components(_stage(img, k, thr, "p"))
    want = ARR[f"{img}_{k}_{thr}_plab"].astype(np.int64)
    assert n == want.max()
    assert np.array_equal(labels, want)


@pytest.mark.parametrize("case", STAGE_CASES, ids=_case_id)
def test_uni_tracks_from_mathematica_selection(case):
    img, k, thr = case
    n = 0
    for ms, mf in SELECTIONS:
        ptag = f"{thr}_s{ms}_f{mf}"
        if ptag in META["tracks"][img]:
            sel = _stage(img, k, thr, f"s{ms}_f{mf}_selL")
            assert _uni_tracks(sel, mf) == _mma_tracks(img, ptag, k), ptag
            n += 1
    assert n >= 3


@pytest.mark.parametrize("case", STAGE_CASES, ids=_case_id)
def test_process_segmentation_uni_with_mathematica_pruning(case, monkeypatch):
    """process_segmentation_uni from Mathematica's binarized net output, with Mathematica's
    Pruning result plugged into the prune_skeleton seam."""
    img, k, thr = case

    def pruning(skeleton, n):
        assert n == 2
        assert np.array_equal(skeleton, _stage(img, k, thr, "t2"))
        return _stage(img, k, thr, "p")

    monkeypatch.setattr(morph, "prune_skeleton", pruning)
    b0 = _stage(img, k, thr, "b0").astype(np.float32)
    for ms, mf in SELECTIONS:
        sel = process_segmentation_uni(b0, _dims(img), 0.5, ms, mf)
        assert np.array_equal(sel, _stage(img, k, thr, f"s{ms}_f{mf}_selL")), (ms, mf)


@pytest.mark.skipif(
    _model("unidirectional_seg.onnx") is None, reason="unidirectional_seg.onnx not found"
)
@pytest.mark.parametrize("img", IMAGES)
def test_uni_end_to_end_matches_mathematica(img):
    """UniKymoButlerSegment + UniKymoButlerTrack from the image file: identical tracks, in
    Mathematica's order, for all 7 parameter sets, through the production pipeline as is."""
    from kymobutler.models.weights import _load_onnx_model
    from kymobutler.segmentation import segment_unidirectional
    from kymobutler.tracking import track_unidirectional

    net = _load_onnx_model("uninet", _model("unidirectional_seg.onnx"), "cpu")
    _, raw, pre, pred = segment_unidirectional(DATA / f"{img}.png", net, "cpu")
    assert pred["ant"].shape == tuple(META["shapes"][img]["net"])
    for ptag in META["tracks"][img]:
        thr, ms, mf = _parse(ptag)
        ant, ret = track_unidirectional(pred, pre.shape, thr, ms, mf)
        assert [t.points for t in ant] == _mma_tracks(img, ptag, "ant"), ptag
        assert [t.points for t in ret] == _mma_tracks(img, ptag, "ret"), ptag


# ------------------------------------------------------------------------------ classifier
@pytest.mark.parametrize("img", IMAGES)
def test_classifier_input_matches_mathematica(img):
    x = classifier_input(DATA / f"{img}.png")
    assert x.shape == (64, 64) and x.dtype == np.float32
    assert np.array_equal(np.rint(x.astype(np.float64) * 255).astype(np.uint8), ARR[f"{img}_cls"])


@pytest.mark.skipif(_model("classifier.onnx") is None, reason="classifier.onnx not found")
def test_classifier_matches_mathematica_web_form():
    from kymobutler.models.weights import _load_onnx_model

    net = _load_onnx_model("classnet", _model("classifier.onnx"), "cpu")
    for img in IMAGES:
        p = classify_probabilities(classifier_input(DATA / f"{img}.png"), net)
        want = META["classifier"][img]
        assert np.abs(p - want).max() < 1e-5, img
        expected = "unidirectional" if want[0] >= want[1] else "bidirectional"
        assert classify_direction(DATA / f"{img}.png", net) == expected
    # random crops of the public images, both polarities: same decision as Mathematica
    pixels = {img: read_image(DATA / f"{img}.png") for img in IMAGES}
    n = 0
    for w in META["windows"]:
        px, levels = pixels[w["tag"]]
        crop = px[w["r0"] : w["r0"] + w["h"], w["c0"] : w["c0"] + w["w"]]
        for x, want in ((crop, w["p_uni"]), (1.0 - crop, w["p_uni_negated"])):
            if x.max() == x.min():
                continue
            x64 = classifier_input_from_pixels(np.rint(x * levels) / levels, levels)
            p = classify_probabilities(x64, net)
            assert (p[0] >= 0.5) == (want >= 0.5), w["id"]
            assert abs(p[0] - want) < 2e-2, w["id"]
            n += 1
    assert n >= 90
