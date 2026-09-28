"""Image import and preprocessing of every pixel format against Mathematica 14.3.

tests/data/formats holds small synthetic image files of every kind (8/16-bit grey, colour,
palette, alpha, signed/unsigned 8/16/32-bit integer, 32/64-bit float, CMYK, 1-bit, BMP, GIF,
JPEG, multi-frame, ...). tests/data/reference/formats_mma.{npz,json} hold what Mathematica made
of each one (see the "generator" field): ``ImageData@Import``, ``rawkym``, the classifier input
``ImageResize[rawkym, {64, 64}]`` and, for the kymograph-like files, ``negkym`` and the net input.
Everything must be reproduced exactly (bit for bit), except the float64 classifier input of
Real64 images, which is compared as the float32 values the classifier sees.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from kymobutler.classification import classifier_input
from kymobutler.preprocessing import (
    UnsupportedImageFormat,
    _cmyk_to_rgb,
    import_image,
    load_and_preprocess_mma,
    load_grayscale,
    read_image,
    resize_to_multiple_of_16,
)

DATA = Path(__file__).parent / "data"
FORMATS = DATA / "formats"
META = json.loads((DATA / "reference" / "formats_mma.json").read_text())["files"]
REF = np.load(DATA / "reference" / "formats_mma.npz")
LEVELS = {"Bit": 1, "Byte": 255, "Bit16": 65535, "Real32": None, "Real64": None}

READABLE = sorted(n for n, m in META.items() if "type" in m and n != "f16.tif")
REJECTED = sorted(n for n, m in META.items() if "mathematica" in m)


def _values(key: str, image_type: str) -> np.ndarray:
    """A stored Mathematica array as float64 values (integer types were stored as levels)."""
    a = REF[key]
    lv = LEVELS[image_type]
    return a.astype(np.float64) / lv if lv else a.astype(np.float64)


@pytest.mark.parametrize("name", READABLE)
def test_import_matches_mathematica(name):
    m = META[name]
    pixels, levels, image_type = import_image(FORMATS / name)
    assert image_type == m["type"]
    assert levels == LEVELS[m["type"]]
    assert read_image(FORMATS / name)[1] == levels
    if f"{name}|data" not in REF:
        return
    data = _values(f"{name}|data", m["type"])
    if m["colorspace"] == "CMYK":  # RemoveAlphaChannel keeps CMYK; every later step sees RGB
        want = _cmyk_to_rgb(data, levels)
    elif m["channels"] in (2, 4):  # RemoveAlphaChannel
        want = data[..., 0] if m["channels"] == 2 else data[..., :3]
    else:
        want = data
    assert pixels.shape == want.shape
    assert np.array_equal(pixels, want)


@pytest.mark.parametrize("name", [n for n in READABLE if f"{n}|raw" in REF])
def test_rawkym_matches_mathematica(name):
    m = META[name]
    raw, levels = load_grayscale(FORMATS / name)
    assert levels == LEVELS[m["raw_type"]]
    assert np.array_equal(raw, _values(f"{name}|raw", m["raw_type"]))


@pytest.mark.parametrize("name", [n for n in READABLE if f"{n}|cls" in REF])
def test_classifier_input_matches_mathematica(name):
    m = META[name]
    x = classifier_input(FORMATS / name)
    assert x.dtype == np.float32 and x.shape == (64, 64)
    want = _values(f"{name}|cls", m["raw_type"]).astype(np.float32)
    assert np.array_equal(x, want)


@pytest.mark.parametrize("name", [n for n in READABLE if f"{n}|neg" in REF])
def test_negkym_and_net_input_match_mathematica(name):
    neg, _, negated, _ = load_and_preprocess_mma(FORMATS / name)
    assert np.array_equal(neg, REF[f"{name}|neg"].astype(np.float64))  # float32 values
    if "negated" in META[name]:
        assert negated == META[name]["negated"]
    if f"{name}|netin" in REF:
        assert np.array_equal(resize_to_multiple_of_16(neg), REF[f"{name}|netin"])


@pytest.mark.parametrize("name", REJECTED)
def test_files_mathematica_cannot_import_are_rejected(name):
    """Multi-frame files (Import returns a list) and 1-bit TIFFs without a BitsPerSample tag
    (Import fails): a clear error, a NotImplementedError subclass, naming the file."""
    with pytest.raises(UnsupportedImageFormat, match=rf"^{name}: ") as err:
        import_image(FORMATS / name)
    assert isinstance(err.value, NotImplementedError)
    if META[name]["mathematica"].startswith("Import returns"):
        assert "images (frames/pages)" in str(err.value)
    else:
        assert "BitsPerSample" in str(err.value)


def test_half_precision_tiff_is_rejected():
    """Mathematica misreads a float16 TIFF as 16-bit integers (a meaningless Bit16 image):
    KymoButler raises instead of reproducing that."""
    assert META["f16.tif"]["type"] == "Bit16"
    with pytest.raises(UnsupportedImageFormat, match="half-precision"):
        import_image(FORMATS / "f16.tif")


def test_not_an_image_raises_pil_error(tmp_path):
    from PIL import UnidentifiedImageError

    (tmp_path / "x.tif").write_bytes(b"not an image")
    with pytest.raises(UnidentifiedImageError):
        import_image(tmp_path / "x.tif")


def test_signed_and_32bit_integer_mapping():
    """int16 -> Bit16 (v + 32768)/65535; int32 -> Real32 float32(v/(2^31 - 1))."""
    tifffile = pytest.importorskip("tifffile")
    a16 = tifffile.imread(FORMATS / "i16_edges.tif").astype(np.int64)
    px, lv, it = import_image(FORMATS / "i16_edges.tif")
    assert (it, lv) == ("Bit16", 65535) and np.array_equal(px * 65535, a16 + 32768)
    a32 = tifffile.imread(FORMATS / "i32_big.tif").astype(np.float64)
    px, lv, it = import_image(FORMATS / "i32_big.tif")
    assert (it, lv) == ("Real32", None)
    assert np.array_equal(px, (a32 / 2147483647.0).astype(np.float32).astype(np.float64))


def test_grey_bmp_is_rgb_and_1bit_bmp_is_bit():
    px, _, it = import_image(FORMATS / "l.bmp")
    assert it == "Byte" and px.shape == (7, 11, 3)
    assert np.array_equal(px[..., 0], px[..., 1]) and np.array_equal(px[..., 0], px[..., 2])
    px, lv, it = import_image(FORMATS / "b1.bmp")
    assert (it, lv) == ("Bit", 1) and px.shape == (7, 11)
