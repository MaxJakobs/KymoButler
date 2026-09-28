"""Tests for kymobutler.io_utils: summary.csv, the track coordinates CSV and the overlay, against
Mathematica 14.3 (tests/data/reference/pproc_mma.json, made by scripts/make_pproc_reference.py).

The fixture holds, for the Mathematica tracks of the public test images and for synthetic tracks
(steps back in time, time gaps): the exported CSV text of pproc (the web form's summary.csv),
pprocLocal (the Mathematica package's) and ConvertToCSV (trackCoordinates.csv) over 13 pixel size
pairs, ToString and CSV cell forms of numbers, the overlay's line image ("coloredlines" with one
colour per track number), a whole overlay, and the overlay bytes of 16-bit and Real32 kymographs.
"""

import base64
import json
import math
import zlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from kymobutler.io_utils import (
    SUMMARY_COLUMNS,
    create_overlay,
    kymograph_bytes,
    mma_csv,
    mma_real,
    mma_tostring,
    save_statistics_csv,
    save_tracks_csv,
    save_tracks_json,
    summary_rows,
    track_colors,
    track_coordinate_rows,
    track_line_pixels,
)
from kymobutler.postprocessing import postprocess
from kymobutler.preprocessing import load_grayscale
from kymobutler.tracking import Track

DATA = Path(__file__).parent / "data"
REF = json.loads((DATA / "reference" / "pproc_mma.json").read_text())
HDR = REF["header"]
COORD_KEYS = sorted(k for k, v in REF["csv"].items() if "coords" in v)


def _num(s: str):
    """A number as typed in the fixture: "1" -> 1 (exact), "0.5" -> 0.5, "1.*^-6" -> 1e-06."""
    return float(s.replace("*^", "e")) if ("." in s or "*^" in s) else int(s)


def _zero_based(tracks):
    return [[(t - 1, x - 1) for t, x in trk] for trk in tracks]


def _unpack(blob: str, shape) -> np.ndarray:
    return np.frombuffer(zlib.decompress(base64.b64decode(blob)), np.uint8).reshape(shape)


def _case(key):
    name, ts, xs = key.split("|")
    return _zero_based(REF["sets"][name]), _num(ts), _num(xs)


# ------------------------------------------------------------------------------ summary.csv


@pytest.mark.parametrize("key", sorted(REF["csv"]))
def test_summary_csv_matches_mathematica(key, tmp_path):
    tracks, tsz, xsz = _case(key)
    stats = postprocess(tracks, tsz, xsz)
    out = tmp_path / "summary.csv"
    save_statistics_csv(stats, out, tsz, xsz, min_frames=HDR["minT"], min_size=HDR["minSz"],
                        threshold=_num(HDR["thr"]), version=HDR["version"])
    assert out.read_bytes().decode() == REF["csv"][key]["summary"]  # the web form's pproc
    save_statistics_csv(stats, out, tsz, xsz)
    assert out.read_bytes().decode() == REF["csv"][key]["local"]  # the package's pprocLocal


def test_summary_header_rows():
    rows = summary_rows([], 0.5, 1, min_frames=3, min_size=10, threshold=0.2, version="9.9")
    assert rows == [
        ["KymoButler Version 9.9 Summary", "pixelsize time= 0.5 sec", "pixelsize space= 1 um",
         "minimum frames= 3", "minimum obj size= 10", "threshold= 0.2"],
        list(SUMMARY_COLUMNS),
    ]
    assert summary_rows([], 1.0, 0.065)[0] == ["pixelsize time= 1 sec", "pixelsize space= 0.065 um"]
    from kymobutler import __version__

    first = summary_rows([], min_frames=1, min_size=1, threshold=1)[0]
    assert first[0] == f"KymoButler Version {__version__} Summary"
    assert first[3:] == ["minimum frames= 1", "minimum obj size= 1", "threshold= 1"]
    assert summary_rows([], min_frames=3.0, min_size=10.0, threshold=0.2)[0][3:] == [
        "minimum frames= 3", "minimum obj size= 10", "threshold= 0.2"]


def test_summary_header_arguments_go_together():
    with pytest.raises(ValueError, match="together"):
        summary_rows([], min_frames=3)
    with pytest.raises(ValueError, match="version"):
        summary_rows([], version="1.0")


def test_summary_without_tracks_has_the_two_header_rows(tmp_path):
    save_statistics_csv([], tmp_path / "s.csv", 1, 1)
    assert (tmp_path / "s.csv").read_text() == (
        '"pixelsize time= 1 sec","pixelsize space= 1 um"\n'
        + ",".join(f'"{c}"' for c in SUMMARY_COLUMNS) + "\n"
    )


# ------------------------------------------------------------------------------ track coordinates


@pytest.mark.parametrize("key", COORD_KEYS)
def test_track_coordinates_csv_matches_mathematica(key, tmp_path):
    tracks, tsz, xsz = _case(key)
    out = tmp_path / "tracks.csv"
    save_tracks_csv([Track(points=t) for t in tracks], out, tsz, xsz)
    assert out.read_bytes().decode() == REF["csv"][key]["coords"]


def test_track_coordinates_keep_point_order():
    # a step back in time stays where it is (ConvertToCSV writes the points in their order)
    rows = track_coordinate_rows([[(9, 4), (8, 6), (10, 7)], [(0, 0), (1, 1)]], 1, 1)
    assert rows[0] == ["t [pixel] #1", "x [pixel] #1", "t [pixel] #2", "x [pixel] #2"]
    assert [r[:2] for r in rows[1:]] == [[10, 5], [9, 7], [11, 8]]
    assert rows[3][2:] == [None, None]
    assert mma_csv(rows).splitlines()[3] == '11,8,"Null","Null"'


def test_track_coordinates_without_tracks(tmp_path):
    save_tracks_csv([], tmp_path / "t.csv")
    assert (tmp_path / "t.csv").read_text() == '"None"\n'


def test_track_coordinates_bad_pixel_size():
    with pytest.raises(ValueError, match="positive"):
        track_coordinate_rows([[(0, 0), (1, 1)]], 0, 1)


def test_tracks_json_keeps_point_order(tmp_path):
    save_tracks_json([Track(points=[(np.int64(9), 4), (8, 6), (10, 7)])], tmp_path / "t.json")
    data = json.loads((tmp_path / "t.json").read_text())
    points = [{"time": 9, "space": 4}, {"time": 8, "space": 6}, {"time": 10, "space": 7}]
    assert data == {"tracks": [{"id": 1, "points": points}]}


# ------------------------------------------------------------------------------ number forms


@pytest.mark.parametrize("typed,want", REF["tostring"])
def test_tostring_matches_mathematica(typed, want):
    assert mma_tostring(_num(typed)) == want


def _near_7th_digit_tie(x: float) -> bool:
    x = abs(x)
    q = Fraction(x) / Fraction(10) ** (math.floor(math.log10(x)) - 6)  # 7 digits before the point
    return abs(float(q - (math.floor(q / 10) * 10 + 5))) / float(q) < 1e-14


def test_tostring_of_machine_reals():
    """Exact except for reals at a rounding tie of the 7th significant digit (typed with a
    final 5 there), where Mathematica rounds about one in ten the other way (see mma_tostring)."""
    far = near = near_bad = 0
    for inputform, want in REF["tostring_reals"]:
        x = float(inputform.replace("*^", "e"))
        got = mma_tostring(x)
        if not _near_7th_digit_tie(x):
            far += 1
            assert got == want, inputform
            continue
        near += 1
        if got != want:
            near_bad += 1
            # the same 6 digits but for one unit in the last
            g, w = (float(s.split("\n")[-1].replace(" 10", "")) for s in (got, want))
            unit = 10 ** (math.floor(math.log10(abs(w))) - 5)
            assert abs(g - w) <= 1.01 * unit, (inputform, got, want)
    assert far >= 250 and near >= 300
    assert near_bad <= 0.15 * near


@pytest.mark.parametrize("typed,ragged,typed_col,with_null", REF["cells"])
def test_csv_cells_match_mathematica(typed, ragged, typed_col, with_null):
    x = _num(typed)
    assert mma_csv([["a", "b"], [x]]) == ragged  # ragged table: InputForm
    assert mma_csv([["a"], [x]]) == typed_col  # a column of one type below a header
    assert mma_csv([["a"], [x], [None]]) == with_null  # a column with a Null: text


def test_mma_real():
    assert [mma_real(v) for v in (1.0, -0.5, 0.0, 1e6, 0.30000000000000004, 1.234e-6)] == [
        "1.", "-0.5", "0.", "1.e6", "0.30000000000000004", "1.234e-6"]
    with pytest.raises(ValueError):
        mma_real(float("nan"))


def test_csv_rejects_unknown_cells():
    with pytest.raises(TypeError):
        mma_csv([[object()]])


# ------------------------------------------------------------------------------ overlay


def _line_ids(tracks, shape) -> np.ndarray:
    ids = np.zeros(shape, np.uint8)
    for k, t in enumerate(tracks, 1):  # later tracks are drawn over earlier ones
        for r, c in track_line_pixels(t, shape):
            ids[r, c] = k
    return ids


@pytest.mark.parametrize("name", sorted(REF["lines"]))
def test_line_pixels_match_mathematica(name):
    """Pixel for pixel, including where a line passes exactly through a pixel corner (the
    synthetic tracks, with long time gaps and steps back in time, have many such corners:
    Mathematica's float32 DDA decides them)."""
    case = REF["lines"][name]
    want = _unpack(case["ids"], case["shape"])
    tracks = _zero_based(REF["sets"][case["set"]])
    assert np.array_equal(_line_ids(tracks, want.shape), want)


def test_line_pixels_float32_corner_ties():
    # Segment (10, 22) -> (16, 11) (0-based): the line passes exactly through the corner of
    # pixels (12, 17) and (13, 17) at the centre of column 17; exact arithmetic gives row 13,
    # Mathematica's single-precision accumulation of the slope -6/11 lands just above 13
    # and lights row 12 (measured).
    px = track_line_pixels([(10, 22), (16, 11)], (40, 40))
    assert (12, 17) in px and (13, 17) not in px
    assert len(px) == 12  # one per column 12..23 (corner columns), no more


def test_line_pixels_of_short_tracks():
    assert track_line_pixels([(5, 7)], (20, 20)) == [(4, 8)]  # a single point: (r - 1, c + 1)
    assert track_line_pixels([], (20, 20)) == []
    assert track_line_pixels([(0, 0), (1, 0)], (20, 20)) == [(0, 1), (1, 1)]
    assert track_line_pixels([(3, 19), (4, 19)], (20, 20)) == []  # last column: off the image


def test_line_pixels_of_repeated_points():
    # Measured on Mathematica 14.3 (coloredlines of {{5,5},{5,5},{9,12}} etc., 1-based): a
    # zero-length segment lights nothing; a line of one repeated point is drawn like one point.
    shape = (20, 20)
    line = track_line_pixels([(4, 4), (8, 11)], shape)
    assert sorted(set(line)) == [(4, 5), (4, 6), (5, 7), (6, 8), (6, 9), (7, 10), (7, 11), (8, 12)]
    assert track_line_pixels([(4, 4), (4, 4), (8, 11)], shape) == line
    bent = [(4, 4), (8, 11), (11, 2)]
    assert track_line_pixels([(4, 4), (8, 11), (8, 11), (11, 2)], shape) == track_line_pixels(
        bent, shape)
    assert track_line_pixels([(4, 4), (4, 4)], shape) == [(3, 5)] == track_line_pixels([(4, 4)], shape)


def test_overlay_matches_mathematica(tmp_path):
    case = REF["overlays"]["bitest"]
    want = _unpack(case["pixels"], case["shape"])  # RGBA, Mathematica's colours
    raw, _ = load_grayscale(DATA / "bitest.png")
    tracks = [Track(points=t) for t in _zero_based(REF["sets"][case["set"]])]
    got = create_overlay(raw, tracks, tmp_path / "o.png")
    ref_lines = REF["lines"]["bitest_s10_f10"]
    lines = _unpack(ref_lines["ids"], ref_lines["shape"]) > 0
    assert np.array_equal(got[~lines], want[..., :3][~lines])  # the kymograph under the lines
    grey = np.repeat(kymograph_bytes(raw)[..., None], 3, 2)
    assert np.array_equal(np.any(got != grey, -1), lines)
    colours = {tuple(got[r, c]) for r, c in np.argwhere(lines)}
    assert colours <= set(track_colors(len(tracks)))
    png = np.asarray(Image.open(tmp_path / "o.png"))
    assert png.shape == tuple(case["shape"]) and png.dtype == np.uint8  # RGBA like Mathematica's
    assert np.array_equal(png[..., :3], got) and (png[..., 3] == 255).all()


@pytest.mark.parametrize("name", ["bit16", "real32"])
def test_overlay_bytes_of_16bit_and_real_kymographs(name):
    case = REF["kymtypes"][name]
    want = _unpack(case["bytes"], case["shape"])
    if name == "bit16":
        kym = (np.arange(65536).reshape(256, 256) / 65535).astype(np.float32)
    else:
        values = [float(v.replace("*^", "e")) for v in case["values"]]
        kym = np.array(values, np.float32).reshape(case["shape"])
    assert np.array_equal(kymograph_bytes(kym), want)
    assert case["alpha"] == [255]


def test_overlay_without_tracks_is_the_grey_kymograph(tmp_path):
    raw = np.linspace(0, 1, 20 * 30, dtype=np.float32).reshape(20, 30)
    rgb = create_overlay(raw, [], tmp_path / "o.png")
    png = Image.open(tmp_path / "o.png")
    assert png.mode == "L"
    assert np.array_equal(np.asarray(png), kymograph_bytes(raw))
    assert np.array_equal(rgb[..., 1], kymograph_bytes(raw))


def test_overlay_lines_follow_the_point_order():
    """A step back in time is drawn back and forth: the line pixels are those of each segment
    in the track's point order (the synthetic reference tracks, which Mathematica drew, have
    such steps at the start, in the middle and at the end)."""
    pts = [(9, 4), (8, 6), (10, 7)]
    shape = (20, 20)
    per_segment = track_line_pixels(pts[:2], shape) + track_line_pixels(pts[1:], shape)
    assert track_line_pixels(pts, shape) == per_segment
    syn = REF["sets"]["synthetic"]
    assert any(b[0] < a[0] for t in syn for a, b in zip(t, t[1:]))


def test_overlay_rejects_values_outside_unit_range():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        create_overlay(np.full((4, 4), 2.0), [])
    with pytest.raises(ValueError, match="grayscale"):
        create_overlay(np.zeros((4, 4, 3)), [])
