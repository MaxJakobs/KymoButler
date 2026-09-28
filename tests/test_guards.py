"""Inputs Mathematica's KymoButler cannot analyse: they must fail early and understandably.

- kymographs with a side of 8 px or less (both networks run at 16*Round[d/16] = 0 px there);
- bidirectional tracking with min_size <= 1 and min_frames <= 1 on a kymograph whose skeleton
  keeps one-pixel pieces and whose tracks overlap (BiKymoButlerTrack's result is undefined:
  Pick::incomp); well defined, and so analysed, as soon as one of the two is 2 or more.
"""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import label

from kymobutler.morphology import process_segmentation_bi
from kymobutler.segmentation import segment_bidirectional, segment_unidirectional
from kymobutler.tracking import track_paths_bidirectional

REF = Path(__file__).parent / "data" / "reference"
_MODEL_DIRS = (Path(__file__).parent.parent / "models", Path.home() / ".kymobutler" / "models")
_DECNET = next((d / "decision_module.onnx" for d in _MODEL_DIRS if (d / "decision_module.onnx").exists()), None)


@pytest.mark.parametrize("h, w", [(8, 50), (50, 8), (8, 8), (1, 1), (3, 200)])
def test_tiny_image_files_fail_alike_before_segmentation(tmp_path, h, w):
    f = tmp_path / f"tiny_{h}x{w}.png"
    Image.fromarray((np.random.default_rng(h * w).random((h, w)) * 255).astype(np.uint8)).save(f)
    msg = f"^KymoButler needs an image of at least 9 x 9 pixels; this image is {w} x {h}$"
    # raised before the network is used: no model needed
    with pytest.raises(ValueError, match=msg):
        segment_bidirectional(f, None)
    with pytest.raises(ValueError, match=msg):
        segment_unidirectional(f, None)


def _pieces_of_one_pixel(paths):
    lab, _ = label(paths, np.ones((3, 3)))
    return int((np.bincount(lab.ravel())[1:] == 1).sum())


@pytest.mark.skipif(_DECNET is None, reason="decision_module.onnx not found")
def test_one_pixel_pieces_with_overlapping_tracks_need_min_size_or_min_frames_above_one():
    from kymobutler.models.weights import _load_onnx_model

    decnet = _load_onnx_model("decnet", _DECNET, "cpu")
    a = np.load(REF / "bitest_mma.npz")  # Mathematica's net output and kympreproc for bitest.png
    kym = a["kympreproc"].astype(np.float32)
    paths = process_segmentation_bi(a["pred"], kym.shape, 0.2, 1, 1)
    assert _pieces_of_one_pixel(paths) == 1
    with pytest.raises(
        ValueError,
        match=r"^KymoButler cannot analyse this kymograph with min_size = 1 and min_frames = 1: "
        r".* Set min_size \(minimum object size, in pixels\) or min_frames \(minimum frames per "
        r"track\) to 2 or more\.$",
    ):
        track_paths_bidirectional(paths, kym, 0.5, decnet, 1, 1)
    # min_frames = 2 removes the one-pixel piece: well defined again
    paths = process_segmentation_bi(a["pred"], kym.shape, 0.2, 1, 2)
    assert _pieces_of_one_pixel(paths) == 0
    assert len(track_paths_bidirectional(paths, kym, 0.5, decnet, 1, 2)) == 15


@pytest.mark.skipif(_DECNET is None, reason="decision_module.onnx not found")
def test_one_pixel_pieces_without_overlapping_tracks_are_fine():
    # unitest2's bidirectional skeleton at min_size = min_frames = 1 keeps two one-pixel pieces,
    # but none of its tracks overlap, so Mathematica's result is well defined
    from kymobutler.models.weights import _load_onnx_model

    decnet = _load_onnx_model("decnet", _DECNET, "cpu")
    a = np.load(REF / "unitest2_mma.npz")
    kym = a["kympreproc"].astype(np.float32)
    paths = process_segmentation_bi(a["pred"], kym.shape, 0.2, 1, 1)
    assert _pieces_of_one_pixel(paths) == 2
    assert len(track_paths_bidirectional(paths, kym, 0.5, decnet, 1, 1)) == 10
