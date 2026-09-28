#!/usr/bin/env python3
"""Compare Python tracks with Mathematica's on the public test images, track for track.

Usage:
    python scripts/validate_tracks.py [--model-dir DIR] [--out DIR]

Runs the production pipeline from the image file (segment_* at the network's resolution, then
track_*) and compares with Mathematica 14.3's tracks in tests/data/reference:
    bidirectional   manifest.json cases: <img>_mma_tracks_t<thr>_s<minSz>_f<minFr>.json
                    (BiKymoButler)
    unidirectional  uni_mma.json: tracks[img]["t<100 thr>_s<minSz>_f<minFr>"]["ant"|"ret"]
                    (UniKymoButlerTrack)
Tracks are 1-based {time, space}, in Mathematica's order, and each track's points in
Mathematica's order (first appearance: a bidirectional track can step back a frame). A case
matches only if Python has the same tracks in the same order with the same points in the same
order; otherwise the report says how they differ, and a side-by-side overlay (Python |
Mathematica, drawn like KymoButler's overlay by io_utils.create_overlay) goes to --out.
tests/test_mma_fidelity.py and tests/test_uni_fidelity.py assert the same, without the report.

Exits with status 1 if any case differs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "tests" / "data"
REF = DATA / "reference"


def describe(py: list[list[tuple[int, int]]], mma: list[list[tuple[int, int]]]) -> str:
    """How two track lists differ (both 1-based, as lists of (time, space) tuples)."""
    if py == mma:
        return "identical"
    ps, ms = [frozenset(t) for t in py], [frozenset(t) for t in mma]
    if sorted(map(sorted, ps)) == sorted(map(sorted, ms)):
        if sorted(py) == sorted(mma):
            return "same tracks, different track order"
        return "same point sets, different point order within tracks"
    missing = [i + 1 for i, s in enumerate(ms) if s not in ps]
    extra = [i + 1 for i, s in enumerate(ps) if s not in ms]
    return (f"differ: Mathematica tracks not found #{missing}; "
            f"Python tracks not in Mathematica #{extra}")


def save_side_by_side(raw: np.ndarray, py, mma, path: Path) -> None:
    from PIL import Image

    from kymobutler.io_utils import create_overlay
    from kymobutler.tracking import Track

    def ov(tracks):
        return create_overlay(
            raw, [Track(points=[(t - 1, x - 1) for t, x in trk]) for trk in tracks]
        )

    sep = np.full((raw.shape[0], 2, 3), 128, np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate([ov(py), sep, ov(mma)], axis=1)).save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=None, help="model weights (default: the repo's models/)")
    ap.add_argument("--out", type=Path, default=DATA / "comparison",
                    help="overlays of differing cases")
    args = ap.parse_args()

    from kymobutler.models.weights import load_default_models
    from kymobutler.segmentation import segment_bidirectional, segment_unidirectional
    from kymobutler.tracking import track_bidirectional, track_unidirectional

    models = load_default_models(args.model_dir)

    def one_based(tracks):
        return [[(int(t) + 1, int(x) + 1) for t, x in trk.points] for trk in tracks]

    def as_tuples(tracks):
        return [[tuple(p) for p in trk] for trk in tracks]

    results = []  # (case, n_mma, n_py, verdict)

    def report(case, raw, py, mma):
        verdict = describe(py, mma)
        results.append((case, len(mma), len(py), verdict))
        if verdict != "identical":
            save_side_by_side(raw, py, mma, args.out / f"{case}_tracks.png")

    for case in json.loads((REF / "manifest.json").read_text())["cases"]:
        was_negated, raw, pre, pred = segment_bidirectional(REPO / case["image"], models["binet"])
        tracks = track_bidirectional(pred, pre, was_negated, case["threshold"], 0.5,
                                     models["decnet"], case["min_size"], case["min_frames"])
        name = (f"bi_{Path(case['image']).stem}_t{case['threshold']}"
                f"_s{case['min_size']}_f{case['min_frames']}")
        want = json.loads((REF / case["tracks"]).read_text())
        report(name, raw, one_based(tracks), as_tuples(want))

    uni = json.loads((REF / "uni_mma.json").read_text())["tracks"]
    for img, by_params in sorted(uni.items()):
        _, raw, pre, pred = segment_unidirectional(DATA / f"{img}.png", models["uninet"])
        for ptag, want in sorted(by_params.items()):
            t, s, f = ptag.split("_")
            ant, ret = track_unidirectional(
                pred, pre.shape, int(t[1:]) / 100, int(s[1:]), int(f[1:])
            )
            for kind, got in (("ant", ant), ("ret", ret)):
                report(f"uni_{img}_{ptag}_{kind}", raw, one_based(got), as_tuples(want[kind]))

    width = max(len(r[0]) for r in results)
    print(f"{'case':{width}s}  mma   py  verdict")
    for case, n_mma, n_py, verdict in results:
        print(f"{case:{width}s} {n_mma:4d} {n_py:4d}  {verdict}")
    bad = [r for r in results if r[3] != "identical"]
    print(f"\n{len(results) - len(bad)}/{len(results)} cases identical to Mathematica"
          + (f"; overlays of the others in {args.out}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
