"""More image-import cases against Mathematica 14.3: orientation, palette, min-is-white, BigTIFF,
MPO, WebP, CMYK rounding and tiny images.

tests/data/formats_extra holds small synthetic files; tests/data/reference/formats_extra_mma.
{json,npz} hold what Mathematica made of each one (see its "generator" field), in the same layout
as tests/data/reference/formats_mma (tests/test_formats.py). Everything must be reproduced bit
for bit.

What these files pin down (all probed in Mathematica 14.3):

- A TIFF's Orientation tag is applied on every reader path (Pillow raw, LZW, PackBits and JPEG
  compression, tifffile), with 2 and 4 swapped relative to the TIFF convention; 0 and 9 are
  ignored, and so are an XMP orientation and the Orientation of a BigTIFF. Pillow turns TIFFs
  itself when it loads them (the TIFF convention, and from XMP too), which must never leak
  through.
- The Exif orientation of JPEGs, of the first image of an MPO and of WebPs is applied the same
  way; a PNG's is ignored.
- Palette TIFFs are Bit16 RGB images of their 16-bit ColorMap; min-is-white TIFFs are negated
  on their type's grid (1-, 8-, 16-bit, int16, float32, with alpha).
- 16-bit CMYK is converted with single-precision arithmetic (``_cmyk_to_rgb``).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from kymobutler.classification import classifier_input
from kymobutler.preprocessing import (
    UnsupportedImageFormat,
    _cmyk_to_rgb,
    image_adjust,
    import_image,
    load_and_preprocess_mma,
    load_grayscale,
    read_image,
    resize_to_multiple_of_16,
)

DATA = Path(__file__).parent / "data"
FILES = DATA / "formats_extra"
META = json.loads((DATA / "reference" / "formats_extra_mma.json").read_text())["files"]
REF = np.load(DATA / "reference" / "formats_extra_mma.npz")
LEVELS = {"Bit": 1, "Byte": 255, "Bit16": 65535, "Real32": None, "Real64": None}

READABLE = sorted(n for n, m in META.items() if "type" in m)
REJECTED = sorted(n for n, m in META.items() if "mathematica" in m)


def _values(key: str, image_type: str) -> np.ndarray:
    a = REF[key]
    lv = LEVELS[image_type]
    return a.astype(np.float64) / lv if lv else a.astype(np.float64)


@pytest.mark.parametrize("name", READABLE)
def test_import_matches_mathematica(name):
    m = META[name]
    pixels, levels, image_type = import_image(FILES / name)
    assert (image_type, levels) == (m["type"], LEVELS[m["type"]])
    assert read_image(FILES / name)[1] == levels
    data = _values(f"{name}|data", m["type"])
    if m["colorspace"] == "CMYK":
        want = _cmyk_to_rgb(data, levels)
    elif m["channels"] in (2, 4):  # RemoveAlphaChannel
        want = data[..., 0] if m["channels"] == 2 else data[..., :3]
    else:
        want = data
    assert pixels.shape == want.shape
    assert np.array_equal(pixels, want)


@pytest.mark.parametrize("name", READABLE)
def test_rawkym_matches_mathematica(name):
    m = META[name]
    raw, levels = load_grayscale(FILES / name)
    assert levels == LEVELS[m["raw_type"]]
    assert np.array_equal(raw, _values(f"{name}|raw", m["raw_type"]))


@pytest.mark.parametrize("name", [n for n in READABLE if f"{n}|cls" in REF])
def test_classifier_input_matches_mathematica(name):
    x = classifier_input(FILES / name)
    assert x.dtype == np.float32 and x.shape == (64, 64)
    want = _values(f"{name}|cls", META[name]["raw_type"]).astype(np.float32)
    assert np.array_equal(x, want)


@pytest.mark.parametrize("name", [n for n in READABLE if f"{n}|neg" in REF])
def test_negkym_and_net_input_match_mathematica(name):
    neg, _, negated, _ = load_and_preprocess_mma(FILES / name)
    assert np.array_equal(neg, REF[f"{name}|neg"].astype(np.float64))
    assert negated == META[name]["negated"]
    if f"{name}|netin" in REF:
        assert np.array_equal(resize_to_multiple_of_16(neg), REF[f"{name}|netin"])


@pytest.mark.parametrize("name", REJECTED)
def test_files_mathematica_cannot_import_are_rejected(name):
    """A truncated TIFF and one whose header declares a height of 0: Import fails."""
    with pytest.raises(UnsupportedImageFormat, match=rf"^{name}: ") as err:
        import_image(FILES / name)
    assert "corrupt" in str(err.value)


def test_orientation_is_mathematicas_not_pillows():
    """Orientation 2: Mathematica flips top to bottom, where the TIFF convention (and Pillow 11,
    which turns TIFFs as it loads them) mirrors left to right. 6: turned clockwise. A BigTIFF's
    tag is ignored."""
    tifffile = pytest.importorskip("tifffile")
    stored = (tifffile.imread(FILES / "o_i16_2.tif").astype(np.int64) + 32768) / 65535
    assert np.array_equal(read_image(FILES / "o_i16_2.tif")[0], stored[::-1])
    stored = (tifffile.imread(FILES / "o_i16_6.tif").astype(np.int64) + 32768) / 65535
    assert np.array_equal(read_image(FILES / "o_i16_6.tif")[0], np.rot90(stored, -1))
    stored = tifffile.imread(FILES / "o_big16_6.tif") / 65535
    assert np.array_equal(read_image(FILES / "o_big16_6.tif")[0], stored)


def test_cmyk16_rounding_ties():
    """1 - fma(C, 1 - K, K) in single precision: exact on pixels whose (1 - C)(1 - K) is within
    0.004 levels of a rounding tie, where the double-precision product is often off by one."""
    cmyk = REF["cmyk16_ties|cmyk"].astype(np.float64) / 65535
    want = REF["cmyk16_ties|adjust"].astype(np.float64)
    got = image_adjust(_cmyk_to_rgb(cmyk, 65535), 65535) * 65535
    assert np.array_equal(got, want)
    naive = np.round((1 - cmyk[..., :3]) * (1 - cmyk[..., 3:]) * 65535)
    assert (naive != want).mean() > 0.05


def test_8bit_cmyk_is_the_rounded_product():
    c, k = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    cmyk = np.stack([c, c, c, k], -1) / 255.0
    want = np.round((1 - cmyk[..., :3]) * (1 - cmyk[..., 3:]) * 255) / 255
    assert np.array_equal(_cmyk_to_rgb(cmyk, 255), want)




@pytest.mark.parametrize("mode", ["L", "RGB"])
def test_lzw_tiffs_decode_like_libtiff(tmp_path, mode):
    """LZW TIFFs (decoded by tifffile via imagecodecs) give the same pixels as Pillow/libtiff."""
    import tifffile

    rng = np.random.default_rng(3)
    arr = rng.integers(0, 255, (37, 53, 3) if mode == "RGB" else (37, 53)).astype(np.uint8)
    f = tmp_path / "lzw.tif"
    Image.fromarray(arr).save(f, compression="tiff_lzw")
    assert np.array_equal(tifffile.imread(f), np.asarray(Image.open(f)))

