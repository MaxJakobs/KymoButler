#!/usr/bin/env python3
"""Compare the networks' prediction maps with Mathematica's, at the network's resolution.

Usage:
    python scripts/validate_predictions.py [--model-dir DIR] [--out DIR]

``segment_bidirectional``/``segment_unidirectional`` return the prediction maps as the networks
produce them: at the network's input size, 16*Round[dims/16] (KymoButler.wl's
``ImageResize[kym, 16*Round@N[dim/16]]``), not at the image size. Mathematica's references are
at that same resolution, so the maps are compared pixel for pixel; a shape mismatch is an error
(it means the network input differs), never something to resize away.

References (public test images only):
    tests/data/reference/<img>_mma.npz "pred"  Mathematica 14.3, bidirectional network
    models/reference/<name>_pred.mat "pred"    legacy/scripts/export_to_onnx.wls: bidirectional
                                               (bitest_binet) and unidirectional (<img>_ant/_ret)

Writes side-by-side images (Python | Mathematica | 5x |difference|) to --out and exits with
status 1 if any map differs by more than --tol.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "tests" / "data"
REF_NPZ = DATA / "reference"
REF_MAT = REPO / "models" / "reference"


def compare(name: str, pred: np.ndarray, mma: np.ndarray, out_dir: Path | None) -> float:
    """Print the difference statistics of one prediction map; returns the max abs difference."""
    if pred.shape != mma.shape:
        raise ValueError(
            f"{name}: the Python prediction map is {pred.shape} but Mathematica's is {mma.shape}; "
            "the network input sizes differ"
        )
    diff = np.abs(pred.astype(np.float64) - mma.astype(np.float64))
    print(f"  {name:22s} {str(pred.shape):12s} max|diff| {diff.max():.2e}  "
          f"mean|diff| {diff.mean():.2e}  range py [{pred.min():.4f}, {pred.max():.4f}] "
          f"mma [{mma.min():.4f}, {mma.max():.4f}]")
    if out_dir is not None:
        from PIL import Image

        def u8(a):
            return np.round(np.clip(a, 0, 1) * 255).astype(np.uint8)

        h, w = pred.shape
        sep = np.full((h, 2), 128, np.uint8)
        canvas = np.concatenate([u8(pred), sep, u8(mma), sep, u8(np.clip(diff * 5, 0, 1))], axis=1)
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(canvas).save(out_dir / f"{name}_comparison.png")
    return float(diff.max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model-dir", default=None, help="model weights (default: the repo's models/)")
    ap.add_argument("--out", type=Path, default=DATA / "comparison", help="comparison images")
    ap.add_argument("--tol", type=float, default=1e-4, help="largest acceptable |difference|")
    args = ap.parse_args()

    from scipy.io import loadmat

    from kymobutler.models.weights import load_default_models
    from kymobutler.segmentation import segment_bidirectional, segment_unidirectional

    models = load_default_models(args.model_dir)
    bi, uni = {}, {}

    def bi_pred(img):
        if img not in bi:
            bi[img] = segment_bidirectional(DATA / f"{img}.png", models["binet"])[3]
        return bi[img]

    def uni_pred(img):
        if img not in uni:
            uni[img] = segment_unidirectional(DATA / f"{img}.png", models["uninet"])[3]
        return uni[img]

    cases = []  # (name, python map getter, Mathematica map)
    for f in sorted(REF_NPZ.glob("*_mma.npz")):
        ref = np.load(f)
        if "pred" in ref.files:
            img = f.name[: -len("_mma.npz")]
            cases.append((f"{img}_bi_mma14", lambda img=img: bi_pred(img), ref["pred"]))
    if REF_MAT.exists():
        for f in sorted(REF_MAT.glob("*_pred.mat")):
            stem = f.name[: -len("_pred.mat")]
            img, kind = stem.rsplit("_", 1)
            mma = loadmat(f)["pred"]
            if kind == "binet":
                cases.append((f"{img}_bi", lambda img=img: bi_pred(img), mma))
            elif kind in ("ant", "ret"):
                cases.append((f"{img}_{kind}", lambda img=img, kind=kind: uni_pred(img)[kind], mma))
    if not cases:
        print("No Mathematica prediction references found.", file=sys.stderr)
        return 1

    print(f"{len(cases)} prediction maps, Python vs Mathematica (network resolution):")
    worst = {name: compare(name, get(), np.asarray(mma, np.float32), args.out)
             for name, get, mma in cases}
    bad = [n for n, d in worst.items() if d > args.tol]
    print(f"\n{len(worst) - len(bad)}/{len(worst)} within {args.tol:g}"
          + (f"; differ: {', '.join(bad)}" if bad else ""))
    print(f"Comparison images: {args.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
