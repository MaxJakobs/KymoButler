"""Output files: summary.csv, the track coordinates CSV, overlays, debug maps.

The CSVs and the overlay are those of the KymoButler web form and Mathematica package
(``pproc`` in KymoButlerPProc.wl, ``ConvertToCSV`` in KymoButlerDeployment.wl, the overlay of
BiKymoButlerTrack/UniKymoButlerTrack in KymoButler.wl), written byte for byte as Mathematica's
``Export[..., "CSV"]`` writes them. ``save_tracks_json`` is a Python-only convenience format.
"""

from __future__ import annotations

import json
import math
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from numbers import Integral
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from kymobutler.postprocessing import STAT_KEYS, mma_number, track_points
from kymobutler.tracking import Track

SUMMARY_COLUMNS = (
    "Direction",
    "Av frame2frame velocity [um/sec]",
    "track duration [sec]",
    "track total displacement [um]",
    "Start2end velocity [um/sec]",
)

_MACHINE_INT = 2**63 - 1  # larger integers are not machine integers in Mathematica


# --------------------------------------------------------------------------- number formats


def _digits(x: float) -> tuple[str, int]:
    """Shortest round-trip decimal digits of a finite nonzero |x| (no trailing zeros) and the
    decimal exponent of the first digit: 0.0125 -> ("125", -2)."""
    _, digits, exp = Decimal(repr(abs(x))).as_tuple()
    return "".join(map(str, digits)).rstrip("0"), len(digits) + exp - 1


def _plain(d: str, e: int, trailing_dot: bool) -> str:
    """Digits ``d`` with first-digit exponent ``e`` in positional notation."""
    if e < 0:
        return "0." + "0" * (-e - 1) + d
    whole, frac = d[: e + 1].ljust(e + 1, "0"), d[e + 1 :]
    return whole + "." + frac if frac or trailing_dot else whole


@lru_cache(maxsize=65536)
def mma_real(x: float) -> str:
    """A machine real as Mathematica writes it (InputForm; CSV cells outside typed columns):
    ``1.``, ``0.30000000000000004``, ``0.00001``, ``1.e6``, ``1.234e-6``."""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"cannot write the non-finite number {x!r}")
    if x == 0:
        return "0."
    d, e = _digits(x)
    s = _plain(d, e, True) if -5 <= e <= 5 else f"{d[0]}.{d[1:]}e{e}"
    return "-" + s if x < 0 else s


@lru_cache(maxsize=65536)
def _typed_real(x: float) -> str:
    """A cell of an all-real CSV column (Mathematica 14.3): ``1`` for 1., ``0.5``,
    ``0.0000012``, ``1e+10``, ``2.5e-7``."""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"cannot write the non-finite number {x!r}")
    if x == 0:
        return "0"
    d, e = _digits(x)
    if -6 <= e <= 9:
        s = _plain(d, e, False)
    else:
        s = d[0] + ("." + d[1:] if len(d) > 1 else "") + ("e+" if e > 0 else "e-") + str(abs(e))
    return "-" + s if x < 0 else s


@lru_cache(maxsize=65536)
def _string_real(x: float) -> str:
    """A real in a CSV column Mathematica writes as text (one holding Null or mixed types):
    ``1.``, ``0.5``, ``1.e+16``, ``5.e-05``."""
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"cannot write the non-finite number {x!r}")
    s = repr(x + 0.0)
    if "e" in s:
        m, ex = s.split("e")
        return (m if "." in m else m + ".") + "e" + ex
    return s[:-1] if s.endswith(".0") else s


def mma_tostring(x) -> str:
    """``ToString`` of a number (the header cells of summary.csv): integers as they are, reals
    with 6 significant digits (``0.15``, ``1.``, ``4.53179``); beyond 10^6 or below 10^-5
    Mathematica's two-line superscript form (``"     6\\n1. 10"``).

    The 6 digits: the real rounded to 16 significant digits, then to 6 with ties rounded up
    (away from zero). Measured on Mathematica 14.3 with 4,100 reals: exact for all 2,153 that
    are not within 10^-14 (relative) of a tie at the 7th significant digit. Of the 1,947 that
    are, i.e. reals typed with a final 5 as 7th significant digit such as 2866.895,
    Mathematica rounds about one in ten the other way, by no rule found (neither the side of
    the tie the binary value lies on, nor the parity, nor the magnitude), so for such a pixel
    size or threshold the header's last digit can differ. (Correct rounding of the binary value
    would differ on a third of them.) 600 of the measured reals are in tests/data/reference."""
    if isinstance(x, bool):
        raise TypeError(f"not a number: {x!r}")
    if isinstance(x, Integral):
        return str(int(x))
    x = float(x)
    if not math.isfinite(x):
        raise ValueError(f"cannot write the non-finite number {x!r}")
    if x == 0:
        return "0."
    q = Decimal(f"{abs(x):.15e}")  # 16 significant digits, correctly rounded
    e = q.adjusted()
    q = q.scaleb(-e).quantize(Decimal("1.00000"), rounding=ROUND_HALF_UP)
    if q >= 10:
        e += 1
        q = (q / 10).quantize(Decimal("1.00000"), rounding=ROUND_HALF_UP)
    d = format(q, "f").replace(".", "").rstrip("0")
    sign = "-" if x < 0 else ""
    if -5 <= e <= 5:
        return sign + _plain(d, e, True)
    line2 = f"{sign}{d[0]}.{d[1:]} 10"
    return " " * len(line2) + str(e) + "\n" + line2


# --------------------------------------------------------------------------- CSV writer


def _quote(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


def _is_int(v) -> bool:
    return isinstance(v, (Integral, np.integer)) and not isinstance(v, (bool, np.bool_))


def _is_real(v) -> bool:
    return isinstance(v, (float, np.floating))


def _ragged_cell(v) -> str:
    if isinstance(v, str):
        return _quote(v)
    if _is_int(v):
        return str(int(v))
    if _is_real(v):
        return mma_real(float(v))
    raise TypeError(f"cannot write {v!r} to a CSV cell")


def _typed_column(values: list) -> list[str]:
    """Cells of one column of a rectangular table below its header, typed as Mathematica 14.3
    types them: all machine integers -> unquoted; all reals -> unquoted reals; anything else
    (a Null, a mix of integers and reals, strings) -> every cell quoted as text."""
    if values and all(_is_int(v) and abs(int(v)) <= _MACHINE_INT for v in values):
        return [str(int(v)) for v in values]
    if values and all(_is_real(v) for v in values):
        return [_typed_real(float(v)) for v in values]
    out = []
    for v in values:
        if v is None:
            out.append('"Null"')
        elif isinstance(v, str):
            out.append(_quote(v))
        elif _is_int(v):
            out.append(_quote(str(int(v))))
        elif _is_real(v):
            out.append(_quote(_string_real(float(v))))
        else:
            raise TypeError(f"cannot write {v!r} to a CSV cell")
    return out


def mma_csv(rows: Sequence[Sequence]) -> str:
    """``ExportString[rows, "CSV"]`` as Mathematica 14.3 writes it (also what the legacy
    KymoButler service wrote).

    A rectangular table whose first row is all strings is a header plus typed columns (see
    ``_typed_column``); any other table is written cell by cell: strings quoted, integers
    plain, reals in InputForm (``1.``, ``0.40280000000000005``). ``None`` is Mathematica's
    ``Null``."""
    rows = [list(r) for r in rows]
    if not rows:
        return ""
    width = len(rows[0])
    if all(len(r) == width for r in rows) and all(isinstance(v, str) for v in rows[0]):
        cols = [_typed_column([r[j] for r in rows[1:]]) for j in range(width)]
        lines = [",".join(_quote(v) for v in rows[0])]
        lines += [",".join(c[i] for c in cols) for i in range(len(rows) - 1)]
    else:
        lines = [",".join(_ragged_cell(v) for v in r) for r in rows]
    return "".join(line + "\n" for line in lines)


def _write_text(text: str, output_path: str | Path) -> None:
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        f.write(text)


# --------------------------------------------------------------------------- summary.csv


def summary_rows(
    stats: Iterable[dict],
    pixel_time: float = 1.0,
    pixel_space: float = 1.0,
    *,
    min_frames: int | None = None,
    min_size: int | None = None,
    threshold: float | None = None,
    version: str | None = None,
) -> list[list]:
    """The rows of summary.csv (``pproc[...][[2]]`` of KymoButlerPProc.wl) for the per-track
    rows of ``postprocessing.postprocess``.

    With ``min_frames``, ``min_size`` and ``threshold`` the first row is ``pproc``'s (the web
    form's): "KymoButler Version <version> Summary", the pixel sizes, "minimum frames= ",
    "minimum obj size= ", "threshold= "; without them it is ``pprocLocal``'s (the Mathematica
    package's): the two pixel sizes only. Pass all three or none. The column-header row
    follows, then one row per track."""
    given = [p is not None for p in (min_frames, min_size, threshold)]
    if any(given) and not all(given):
        raise ValueError("pass min_frames, min_size and threshold together (or none of them)")
    tsz, xsz = mma_number(pixel_time), mma_number(pixel_space)
    pixels = [
        f"pixelsize time= {mma_tostring(tsz)} sec",
        f"pixelsize space= {mma_tostring(xsz)} um",
    ]
    if all(given):
        if version is None:
            from kymobutler import __version__ as version
        first = [
            f"KymoButler Version {version} Summary",
            *pixels,
            f"minimum frames= {mma_tostring(mma_number(min_frames))}",
            f"minimum obj size= {mma_tostring(mma_number(min_size))}",
            f"threshold= {mma_tostring(mma_number(threshold))}",
        ]
    else:
        if version is not None:
            raise ValueError(
                "the version is only part of the web form's header (pass min_frames etc.)"
            )
        first = pixels
    return [first, list(SUMMARY_COLUMNS)] + [[row[k] for k in STAT_KEYS] for row in stats]


def save_statistics_csv(
    stats: Iterable[dict],
    output_path: str | Path,
    pixel_time: float = 1.0,
    pixel_space: float = 1.0,
    *,
    min_frames: int | None = None,
    min_size: int | None = None,
    threshold: float | None = None,
    version: str | None = None,
) -> None:
    """Write summary.csv exactly as Mathematica exported it (see ``summary_rows``).

    ``stats`` are the rows of ``postprocess`` computed with the same pixel sizes. Without
    tracks the file holds the two header rows only, as Mathematica's did."""
    rows = summary_rows(
        stats, pixel_time, pixel_space,
        min_frames=min_frames, min_size=min_size, threshold=threshold, version=version,
    )
    _write_text(mma_csv(rows), output_path)


# --------------------------------------------------------------------------- track coordinates


def track_coordinate_rows(
    tracks: Sequence[Track | Sequence], pixel_time: float = 1.0, pixel_space: float = 1.0
) -> list[list]:
    """``ConvertToCSV[trks, tsz, xsz]`` (KymoButlerDeployment.wl): a column pair
    "t [pixel] #k", "x [pixel] #k" per track, in track order and in each track's point order,
    holding the 1-based (Mathematica) pixel coordinates times the pixel sizes (the headers
    say "[pixel]" regardless, as in the original); shorter tracks are padded with Null.
    Without tracks: the single cell "None"."""
    tsz, xsz = mma_number(pixel_time), mma_number(pixel_space)
    for name, v in (("pixel_time", tsz), ("pixel_space", xsz)):
        if not (math.isfinite(v) and v > 0):
            raise ValueError(f"{name} must be a positive number, got {v!r}")
    if not tracks:
        return [["None"]]
    pts = [track_points(t) for t in tracks]
    n = max(len(p) for p in pts)
    header, cols = [], []
    for k, p in enumerate(pts, 1):
        header += [f"t [pixel] #{k}", f"x [pixel] #{k}"]
        cols.append([(t + 1) * tsz for t, _ in p] + [None] * (n - len(p)))
        cols.append([(x + 1) * xsz for _, x in p] + [None] * (n - len(p)))
    return [header] + [list(r) for r in zip(*cols)]


def save_tracks_csv(
    tracks: Sequence[Track | Sequence],
    output_path: str | Path,
    pixel_time: float = 1.0,
    pixel_space: float = 1.0,
) -> None:
    """Write the track coordinates CSV (the web form's trackCoordinates.csv) exactly as
    Mathematica exported it; see ``track_coordinate_rows``."""
    _write_text(mma_csv(track_coordinate_rows(tracks, pixel_time, pixel_space)), output_path)


def save_tracks_json(tracks: list[Track], output_path: str | Path) -> None:
    """Save tracks to a JSON file: 0-based pixel coordinates, each track in its point order."""
    path = Path(output_path)
    data = {
        "tracks": [
            {"id": i + 1, "points": [{"time": r, "space": c} for r, c in track_points(track)]}
            for i, track in enumerate(tracks)
        ]
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# --------------------------------------------------------------------------- debug maps


def save_segmentation_debug(
    pred: np.ndarray | dict[str, np.ndarray],
    output_dir: str | Path,
    prefix: str = "",
) -> None:
    """Save raw segmentation predictions for debugging.

    Saves each prediction map (at the network's resolution, as ``segment_*`` return it) as a
    grayscale PNG for visual inspection and as a .npy file for numerical comparison with
    Mathematica's ``pred``.

    Args:
        pred: Prediction array (H, W) float32, or dict with 'ant'/'ret' keys.
        output_dir: Directory to save debug outputs.
        prefix: Filename prefix (e.g., 'bitest', 'unitest').
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if isinstance(pred, dict):
        for key, arr in pred.items():
            _save_pred_arrays(arr, out, f"{prefix}_{key}" if prefix else key)
    else:
        _save_pred_arrays(pred, out, prefix or "pred")


def _save_pred_arrays(arr: np.ndarray, out_dir: Path, name: str) -> None:
    """Save a single prediction array as PNG and .npy."""
    from PIL import Image as PILImage

    np.save(out_dir / f"{name}_pred.npy", arr)
    img_uint8 = np.round(np.clip(arr, 0, 1) * 255).astype(np.uint8)
    PILImage.fromarray(img_uint8).save(out_dir / f"{name}_pred.png")


# --------------------------------------------------------------------------- overlay


def kymograph_bytes(image: np.ndarray) -> np.ndarray:
    """The 8-bit pixels of ``rawkym`` in the overlay (``ImageCompose`` onto the byte-valued
    track lines): ``BitShiftRight[Round[65535 x], 8]``, measured on Mathematica 14.3. That is
    the byte itself for an 8-bit kymograph (x = k/255), the high byte for a 16-bit one
    (x = k/65535), and the same rule for a real-valued one.

    Args:
        image: ``rawkym`` as ``segment_*`` return it: (H, W) in [0, 1].
    """
    a = np.asarray(image, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"expected a grayscale kymograph (H, W), got shape {a.shape}")
    if not np.all(np.isfinite(a)) or a.min() < 0 or a.max() > 1:
        raise ValueError("the kymograph must hold values in [0, 1] (rawkym from segment_*)")
    return (np.round(a * 65535.0).astype(np.int64) >> 8).astype(np.uint8)


def track_colors(n: int, seed: int = 42) -> list[tuple[int, int, int]]:
    """One colour per track. Mathematica draws each track in ``RandomColor[]``, which differs
    on every run, so no colour choice reproduces it; these are seeded (the same on every run)
    and kept bright (channels 50-254) so that every track shows up on the kymograph."""
    rng = np.random.default_rng(seed)
    return [tuple(int(c) for c in rng.integers(50, 255, size=3)) for _ in range(n)]


def track_line_pixels(points: Sequence, shape: tuple[int, int]) -> list[tuple[int, int]]:
    """The (row, column) pixels Mathematica lights for one track's line, clipped to the image,
    segment by segment in the track's point order (a pixel shared by two segments is listed
    twice).

    The overlay draws ``Line`` through the track's points in their order (so a step back in
    time is drawn back and forth), with 1-based ``{t, x}`` shifted to ``{t - 1, x}``, rendered
    without antialiasing and rotated into place (BiKymoButlerTrack/UniKymoButlerTrack). In the
    kymograph's own pixel grid a 0-based point (r, c) lands on the pixel corner (r, c + 1), and
    Mathematica 14.3's rasterizer lights, per segment, a single-precision DDA:

    * the major axis is the rows when the segment is taller than it is wide, else the columns;
    * starting at the end with the smaller major coordinate p0, one pixel per row (column)
      whose centre lies between p0 and half a pixel beyond the other end p1: for the k-th,
      ``floor(m_k)`` in the minor axis, where ``m_0 = p0_minor + slope / 2`` and
      ``m_k = m_(k-1) + slope``, with ``slope = d_minor / d_major``, all in float32 (each
      operation rounded to single precision, which decides the pixel wherever the line passes
      through a pixel corner);
    * a segment between two equal points lights nothing, but a track whose points are all
      the same point (a one-point track) lights the pixel (r - 1, c + 1).

    This reproduces Mathematica's line images pixel for pixel (measured with one colour per
    track): the Mathematica tracks of the public test images and synthetic tracks with steps
    back in time and long time gaps (tests/test_io_utils.py), and the 420 tracks of 14 local
    kymographs. As in Mathematica, lines sit one pixel right of the track, and a track in the
    last column is not visible."""
    h, w = shape
    rows, cols, _ = _line_pixels([points])
    keep = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    return list(zip(rows[keep].tolist(), cols[keep].tolist()))


def _line_pixels(tracks: Sequence[Track | Sequence]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rows, columns and 1-based track numbers of the pixels of every track's line (see
    ``track_line_pixels``), unclipped, in drawing order: by track, segment, then along the
    segment."""
    f32 = np.float32
    p0s, p1s, tids, singles = [], [], [], []
    for k, t in enumerate(tracks, 1):
        p = np.asarray(track_points(t), dtype=np.int64).reshape(-1, 2) + (0, 1)  # pixel corners
        if len(p) and (p == p[0]).all():  # one point (or the same point repeated)
            singles.append((p[0, 0] - 1, p[0, 1], k))
        elif len(p):
            p0s.append(p[:-1])
            p1s.append(p[1:])
            tids.append(np.full(len(p) - 1, k, dtype=np.int64))
    empty = np.zeros(0, dtype=np.int64)
    # (rows, cols, track, segment, step) of each batch of pixels
    parts = [tuple(np.array([v], dtype=np.int64) for v in (r, c, k, -1, 0)) for r, c, k in singles]
    if p0s:
        p0, p1, tid = np.concatenate(p0s), np.concatenate(p1s), np.concatenate(tids)
        seg = np.arange(len(p0))
        keep = np.any(p1 != p0, axis=1)  # a zero-length segment lights nothing
        p0, p1, tid, seg = p0[keep], p1[keep], tid[keep], seg[keep]
    if p0s and len(p0):
        n = np.arange(len(p0))
        d = p1 - p0
        major = (np.abs(d[:, 0]) <= np.abs(d[:, 1])).astype(np.int64)  # 0: rows, 1: columns
        minor = 1 - major
        swap = p1[n, major] < p0[n, major]
        p0, p1 = np.where(swap[:, None], p1, p0), np.where(swap[:, None], p0, p1)
        length = p1[n, major] - p0[n, major]
        slope = (p1[n, minor] - p0[n, minor]).astype(f32) / length.astype(f32)
        m = p0[n, minor].astype(f32) + slope * f32(0.5)
        start = p0[n, major]
        order = np.argsort(-length, kind="stable")  # the segments still drawing come first
        length, slope, m, start = length[order], slope[order], m[order], start[order]
        major, tid, seg = major[order], tid[order], seg[order]
        # active[step]: how many segments (the longest) light a pixel at this step
        active = len(length) - np.searchsorted(length[::-1], np.arange(length[0] + 1), "left")
        for step, a in enumerate(active.tolist()):
            if step:
                m[:a] += slope[:a]  # float32 += float32: rounded to single precision
            lo, mi = start[:a] + step, np.floor(m[:a]).astype(np.int64)
            rows = np.where(major[:a] == 0, lo, mi)
            cols = np.where(major[:a] == 0, mi, lo)
            parts.append((rows, cols, tid[:a], seg[:a], np.full(a, step, dtype=np.int64)))
    if not parts:
        return empty, empty, empty
    rows, cols, tid, seg, step = (np.concatenate(x) for x in zip(*parts))
    order = np.lexsort((step, seg, tid))
    return rows[order], cols[order], tid[order]


def create_overlay(
    image: np.ndarray,
    tracks: Sequence[Track | Sequence],
    output_path: str | Path | None = None,
) -> np.ndarray:
    """The overlay of BiKymoButlerTrack/UniKymoButlerTrack: every track drawn as a coloured
    line on the kymograph, later tracks over earlier ones.

    Args:
        image: ``rawkym`` from ``segment_*``: the grayscale kymograph (H, W) in [0, 1]. (The
            bidirectional overlay composes onto ``ImageAdjust@kym``, which is ``kym`` itself
            because ``rawkym`` is already stretched to [0, 1].)
        tracks: Tracks (or their 0-based (time, space) points) in output order
            (unidirectional: anterograde, then retrograde).
        output_path: Optional PNG path. Written as Mathematica exported the overlay: RGBA
            with an opaque alpha channel when there are tracks; without tracks the grayscale
            kymograph (BiKymoButlerTrack returns ``ImageAdjust@kym`` then).

    Returns:
        The overlay as (H, W, 3) uint8 RGB.

    What differs from Mathematica's file: the colours (see ``track_colors``), and two
    zero-track cases, whose pixel values agree but whose PNG layout does not: Mathematica
    exports a 16-bit kymograph without tracks as a 16-bit PNG (this writes 8 bits), and a
    unidirectional kymograph whose components all fail the minimum-frames test as RGBA (this
    writes grayscale).
    """
    from PIL import Image as PILImage

    gray = kymograph_bytes(image)
    rgb = np.repeat(gray[:, :, None], 3, axis=2)
    h, w = gray.shape
    rows, cols, tid = _line_pixels(tracks)
    keep = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    if keep.any():
        owner = np.zeros(gray.shape, dtype=np.int64)  # the last track drawn over each pixel
        np.maximum.at(owner, (rows[keep], cols[keep]), tid[keep])
        lit = owner > 0
        rgb[lit] = np.array(track_colors(len(tracks)), dtype=np.uint8)[owner[lit] - 1]

    if output_path is not None:
        if tracks:
            alpha = np.full(gray.shape + (1,), 255, dtype=np.uint8)
            PILImage.fromarray(np.concatenate([rgb, alpha], axis=2)).save(output_path)  # RGBA
        else:
            PILImage.fromarray(gray).save(output_path)  # L
    return rgb
