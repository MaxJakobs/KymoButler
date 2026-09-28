#!/usr/bin/env python3
"""Build tests/data/reference/pproc_mma.json: Mathematica's summary.csv, trackCoordinates.csv and
overlay line pixels for public and synthetic tracks.

Usage:
    python scripts/make_pproc_reference.py [--wolframscript PATH]

Track sets (1-based Mathematica {time, space} tracks, in Mathematica's order):
    * the Mathematica tracks of the public test images already in tests/data/reference
      (bitest/unitest2 bidirectional, unitest/unitest2/bitest unidirectional);
    * "synthetic": tracks from a seeded generator (below) with time gaps, steps back in time
      (at the start, in the middle and at the end), no-motion and two-point tracks;
    * "long_diag", "long_steep", "long_shallow": seeded tracks with long segments (up to 700
      frames or 2,500 pixels) on large canvases, many of them exactly diagonal, for the
      overlay's line rasterization (single-precision accumulation along long segments);
    * "empty": no tracks.
Mathematica (scripts/export_pproc_reference.wls) computes the CSVs for every set and a list of
pixel sizes, ToString/CSV forms of assorted numbers (typed, and seeded machine reals: 7-digit
decimals, decimals ending in a 5 at the 7th significant digit, random reals), the overlay's line
image of some sets, and the bytes of the overlay composed onto a 16-bit and a Real32 kymograph.
The line images are stored as the track number of every pixel (0: no line).
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import tempfile
import zlib
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
REF = REPO / "tests" / "data" / "reference"
OUT = REF / "pproc_mma.json"

PIXEL_SIZES = [
    ["1", "1"], ["0.5", "0.1"], ["0.3", "0.7"], ["3", "7"], ["2", "0.1"], ["0.1", "2"],
    ["0.065", "0.16"], ["1.5", "0.2"], ["1000", "0.001"], ["0.001", "1000"],
    ["4.531785", "4.6885"], ["0.25", "0.25"], ["7", "3"],
]
COORDS_FOR = {  # set -> pixel sizes whose trackCoordinates CSV the fixture keeps
    "bi_unitest2_t0.2_s3_f3": PIXEL_SIZES,
    "bi_bitest_t0.2_s10_f10": [["0.5", "0.1"]],
    "uni_unitest2_t20_s3_f3": [["1", "1"], ["0.3", "0.7"]],
    "synthetic": [["1", "1"], ["0.5", "0.1"], ["1000", "0.001"], ["4.531785", "4.6885"]],
    "empty": PIXEL_SIZES,
}
HEADER = {"minT": 3, "minSz": 3, "thr": "0.15", "version": "1.1.4"}
TOSTRING = [
    "0.15", "0.2", "0.1", "1", "1.", "0.05", "0.125", "3", "10", "0.065", "0.16", "4.531785",
    "1000", "0.001", "1.*^-6", "1234567.", "0.0000123456", "2.5", "100000.", "999999.5", "0.3",
    "12.0000001", "0.9999995", "1.*^10", "-0.5", "123456.7",
]
CELLS = [
    "1.", "0.5", "0.", "1000000.", "1234567.5", "0.00001", "0.000001", "1.*^-7",
    "0.30000000000000004", "123456789.125", "1.*^10", "1.5*^15", "1.*^16", "12", "-3", "0.0001",
    "5.*^-5", "99999.99999", "4.1000000000000005", "9999999999.", "0.0000012", "2.5*^-7",
]


def tostring_reals(seed: int = 11) -> list[float]:
    """Machine reals for ToString: 7-significant-digit decimals, decimals whose 7th significant
    digit is a final 5 (rounding ties at 6 digits) and random reals, over many magnitudes."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(150):
        out.append(float(f"{int(rng.integers(10**6, 10**7))}e{int(rng.integers(-12, 0))}"))
    for _ in range(300):
        out.append(float(f"{int(rng.integers(10**5, 10**6))}5e{int(rng.integers(-12, 1))}"))
    out += [float(v) for v in rng.uniform(0.0001, 100, 150)]
    return out


def synthetic_tracks(seed: int = 7) -> list[list[list[int]]]:
    """Tracks with distinct frames per track (as KymoButler's), 1-based, within 150 x 400 px."""
    rng = np.random.default_rng(seed)
    tracks = [
        [[10, 5], [9, 7], [11, 8], [12, 8]],  # starts with a step back in time
        [[20, 50], [21, 52], [23, 55], [22, 56]],  # ends with a step back
        [[5, 5], [6, 9]],  # two points
        [[3, 10], [4, 10], [5, 10]],  # no motion
        [[30, 300], [31, 296], [32, 290], [34, 281], [33, 283]],  # retrograde, gaps, back step
        [[40, 100], [41, 101], [40 + 8, 101], [49, 140]],  # a long gap and a jump
    ]
    while len(tracks) < 40:
        n = int(rng.integers(3, 60))
        steps = rng.choice([1, 1, 1, 1, 1, 1, 1, 2, 3, 5], size=n - 1)
        t = list(np.cumsum(np.concatenate([[int(rng.integers(1, 80))], steps])))
        for _ in range(int(rng.integers(0, 4))):  # swap neighbours: steps back in time
            i = int(rng.integers(0, n - 1))
            t[i], t[i + 1] = t[i + 1], t[i]
        x = [int(rng.integers(20, 380))]
        for _ in range(n - 1):
            step = rng.integers(-5, 6) + (30 if rng.random() < 0.05 else 0)
            x.append(int(np.clip(x[-1] + step, 1, 400)))
        trk = [[int(a), int(b)] for a, b in zip(t, x) if a <= 150]
        if len(trk) >= 2 and trk[-1][0] - trk[0][0] + 1 != 0:
            tracks.append(trk)
    return tracks


def long_segment_sets(seed: int = 424242) -> dict[str, list]:
    """Tracks with long segments (1-based, distinct frames per track): exactly diagonal ones on
    a 300 x 300 canvas, steep ones over 2,000 frames and shallow ones over 2,500 pixels."""
    rng = np.random.default_rng(seed)

    def walk(h, w, n, maxjump, backprob, diagonal=False):
        out = []
        while len(out) < n:
            pts = [[int(rng.integers(1, h // 2)), int(rng.integers(1, w))]]
            for _ in range(int(rng.integers(2, 12)) - 1):
                if diagonal and rng.random() < 0.5:
                    d = int(rng.integers(1, maxjump))
                    dt, dx = d, d * (1 if rng.random() < 0.5 else -1)
                else:
                    dt = int(rng.integers(1, maxjump))
                    dx = int(rng.integers(-maxjump, maxjump + 1))
                if rng.random() < backprob:
                    dt = -1
                t, x = pts[-1][0] + dt, int(np.clip(pts[-1][1] + dx, 1, w))
                if 1 <= t <= h and all(p[0] != t for p in pts):
                    pts.append([t, x])
            if len(pts) >= 2 and pts[-1][0] - pts[0][0] + 1 != 0:
                out.append(pts)
        return out

    shallow = []
    while len(shallow) < 60:
        pts = [[int(rng.integers(1, 140)), int(rng.integers(1, 2500))]]
        for _ in range(int(rng.integers(1, 6))):
            t = pts[-1][0] + int(rng.integers(1, 4))
            if t > 150:
                break
            pts.append([t, int(rng.integers(1, 2500))])
        if len(pts) >= 2:
            shallow.append(pts)
    return {
        "long_diag": walk(300, 300, 80, 60, 0.1, diagonal=True),
        "long_steep": walk(2000, 120, 60, 700, 0.05),
        "long_shallow": shallow,
    }


def public_sets() -> dict[str, list]:
    sets = {}
    bi = ("bitest_mma_tracks_t0.2_s10_f10", "bitest_mma_tracks_t0.2_s3_f3",
          "unitest2_mma_tracks_t0.2_s3_f3")
    for f in bi:
        sets["bi_" + f.replace("_mma_tracks", "")] = json.loads((REF / f"{f}.json").read_text())
    uni = json.loads((REF / "uni_mma.json").read_text())["tracks"]
    uni_cases = (("unitest", "t20_s3_f3"), ("unitest2", "t20_s3_f3"), ("bitest", "t15_s10_f10"))
    for img, key in uni_cases:
        # Flatten[{ant, ret}, 1]
        sets[f"uni_{img}_{key}"] = uni[img][key]["ant"] + uni[img][key]["ret"]
    return sets


def _pack(a: np.ndarray) -> str:
    """uint8 pixels, row-major, zlib-compressed and base64-encoded."""
    return base64.b64encode(zlib.compress(np.ascontiguousarray(a, np.uint8).tobytes(), 9)).decode()


def _decode_lines(png: Path, n_tracks: int) -> np.ndarray:
    from PIL import Image

    a = np.asarray(Image.open(png).convert("RGB")).astype(int)
    lit = a.sum(-1) > 0
    ids = np.where(lit, a[..., 0] - 50, 0)
    k = ids[lit]
    ok = (a[lit][:, 2] == 200) & (k >= 1) & (k <= n_tracks) & (a[lit][:, 1] == 50 + (37 * k) % 150)
    if not ok.all():
        raise RuntimeError(f"{png}: pixels that are not a track colour (antialiasing?)")
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--wolframscript",
                    default="/Applications/Wolfram.app/Contents/MacOS/wolframscript")
    args = ap.parse_args()

    sets = public_sets()
    sets["synthetic"] = synthetic_tracks()
    sets.update(long_segment_sets())
    sets["empty"] = []
    lines = [
        {"name": "bitest_s10_f10", "set": "bi_bitest_t0.2_s10_f10", "dims": [284, 284]},
        {"name": "unitest2_bi", "set": "bi_unitest2_t0.2_s3_f3", "dims": [92, 83]},
        {"name": "unitest_uni", "set": "uni_unitest_t20_s3_f3", "dims": [670, 326]},
        {"name": "synthetic", "set": "synthetic", "dims": [400, 150]},
        {"name": "bitest_s3_f3", "set": "bi_bitest_t0.2_s3_f3", "dims": [284, 284]},
        {"name": "bitest_uni", "set": "uni_bitest_t15_s10_f10", "dims": [284, 284]},
        {"name": "unitest2_uni", "set": "uni_unitest2_t20_s3_f3", "dims": [92, 83]},
        {"name": "long_diag", "set": "long_diag", "dims": [300, 300]},
        {"name": "long_steep", "set": "long_steep", "dims": [120, 2000]},
        {"name": "long_shallow", "set": "long_shallow", "dims": [2500, 150]},
    ]
    overlays = [{"name": "bitest", "set": "bi_bitest_t0.2_s10_f10",
                 "image": str(REPO / "tests/data/bitest.png")}]
    job = {"sets": sets, "pixel_sizes": PIXEL_SIZES, "header": HEADER, "tostring": TOSTRING,
           "tostring_reals": tostring_reals(), "cells": CELLS, "lines": lines, "overlays": overlays}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "in.json").write_text(json.dumps(job))
        wls = str(REPO / "scripts/export_pproc_reference.wls")
        subprocess.run([args.wolframscript, "-file", wls, str(tmp / "in.json"),
                        str(tmp / "out.json"), str(tmp)], check=True)
        res = json.loads((tmp / "out.json").read_text())
        line_ids = {}
        for c in lines:
            ids = _decode_lines(tmp / f"{c['name']}_lines.png", len(sets[c["set"]]))
            line_ids[c["name"]] = {"set": c["set"], "shape": list(ids.shape), "ids": _pack(ids)}
        from PIL import Image

        overlay_px = {}
        for c in overlays:
            a = np.asarray(Image.open(tmp / f"{c['name']}_overlay.png"))
            overlay_px[c["name"]] = {"set": c["set"],
                                     "image": str(Path(c["image"]).relative_to(REPO)),
                                     "shape": list(a.shape), "types": res["overlays"][c["name"]],
                                     "pixels": _pack(a)}
        kymtypes = {}
        for name, meta in res["kymtypes"].items():
            a = np.asarray(Image.open(tmp / f"{name}_kymtype.png"))
            grey = a.ndim == 3 and a.shape[2] == 4 and (a[..., :3] == a[..., :1]).all()
            if not grey:  # the RGBA overlay of a grey kymograph has R = G = B
                raise RuntimeError(f"{name}: unexpected overlay {a.shape} {a.dtype}")
            kymtypes[name] = {**meta, "shape": list(a.shape[:2]), "dtype": str(a.dtype),
                              "bytes": _pack(a[..., 0]),
                              "alpha": sorted(int(v) for v in np.unique(a[..., 3]))}
    for key, v in res["csv"].items():  # keep the (large) coordinate CSVs of a few cases only
        name, ts, xs = key.split("|")
        if [ts, xs] not in COORDS_FOR.get(name, []):
            del v["coords"]
    fixture = {
        "generator": res["generator"],
        "content": (
            "summary/local/coords: ExportString[..., \"CSV\"] of pproc (web form), pprocLocal and "
            "ConvertToCSV for csv keys '<set>|<tsz>|<xsz>' (pixel sizes as typed: integers exact); "
            "tostring: [input, ToString]; tostring_reals: [InputForm, ToString] of machine reals; "
            "cells: [input, ragged-table CSV, typed-column CSV, column-with-Null CSV]; lines: "
            "coloredlines track number per pixel (zlib, uint8, row-major); overlays: the exported "
            "overlay PNG's pixels (zlib, uint8); kymtypes: the grey bytes (zlib, row-major) and "
            "alpha values of the overlay PNG composed onto a Bit16 ramp (values 0..65535 "
            "row-major, 256 x 256) and onto Real32 noise (values: InputForm, row-major)."
        ),
        "header": res["header"],
        "sets": sets,
        "csv": res["csv"],
        "tostring": res["tostring"],
        "tostring_reals": res["tostring_reals"],
        "cells": res["cells"],
        "lines": line_ids,
        "overlays": overlay_px,
        "kymtypes": kymtypes,
    }
    OUT.write_text(json.dumps(fixture, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size // 1024} KiB)")


if __name__ == "__main__":
    main()
