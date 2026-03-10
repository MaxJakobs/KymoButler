"""Wavelet-based kymograph segmentation alternative.

Uses StationaryWaveletTransform instead of neural networks.
Corresponds to WaveletPackage.wl AnalyseKymographBIwavelet (lines 361-427).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pywt
import torch
from scipy.ndimage import binary_dilation
from skimage.morphology import thin, remove_small_objects

from kymobutler.morphology import (
    _filter_components,
    _prune_branches,
    detect_seeds,
    smooth_binary_bi,
)
from kymobutler.preprocessing import load_and_preprocess
from kymobutler.tracking import Track, track_bidirectional


def _wavelet_segmentation(
    image: np.ndarray,
    threshold: float = 0.2,
    min_size: int = 10,
    min_frames: int = 10,
) -> np.ndarray:
    """Segment a kymograph using stationary wavelet transform.

    Pipeline from WaveletPackage.wl lines 372-373:
    1. SWT at level 2
    2. Aggregate detail coefficients from sub-bands {0}, {1}, {2}, {0,0}, {0,2}, {0,1}
    3. Binarize, dilate, prune, remove small, thin, filter

    Args:
        image: Preprocessed grayscale kymograph (H, W) float32.
        threshold: Binarization threshold.
        min_size: Minimum component pixel count.
        min_frames: Minimum temporal span.

    Returns:
        Binary skeleton image (H, W) bool.
    """
    # Pad to even dimensions (required by SWT)
    h, w = image.shape
    pad_h = h % 2
    pad_w = w % 2
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w)), mode="reflect")

    # Stationary Wavelet Transform at level 2
    coeffs = pywt.swt2(image, wavelet="haar", level=2, trim_approx=True)

    # coeffs structure from pywt.swt2 with trim_approx=True:
    # [approx_level2, (cH2, cV2, cD2), (cH1, cV1, cD1)]
    # Mathematica sub-bands {0}, {1}, {2}, {0,0}, {0,2}, {0,1}
    # Mapping: level 1 details = (cH1, cV1, cD1), level 2 details = (cH2, cV2, cD2)
    # {0} = level 1 horizontal (cH1), {1} = level 1 vertical (cV1), {2} = level 1 diagonal (cD1)
    # {0,0} = level 2 horizontal (cH2), {0,1} = level 2 vertical (cV2), {0,2} = level 2 diagonal (cD2)

    approx = coeffs[0]
    detail_l2 = coeffs[1]  # (cH2, cV2, cD2)
    detail_l1 = coeffs[2]  # (cH1, cV1, cD1)

    # Aggregate selected sub-bands
    aggregated = (
        np.abs(detail_l1[0])  # {0}: cH1
        + np.abs(detail_l1[1])  # {1}: cV1
        + np.abs(detail_l1[2])  # {2}: cD1
        + np.abs(detail_l2[0])  # {0,0}: cH2
        + np.abs(detail_l2[2])  # {0,2}: cD2
        + np.abs(detail_l2[1])  # {0,1}: cV2
    )

    # Rescale to [0, 1]
    if aggregated.max() > aggregated.min():
        aggregated = (aggregated - aggregated.min()) / (aggregated.max() - aggregated.min())

    # Remove padding
    aggregated = aggregated[:h, :w]

    # Binarize
    binary = aggregated > threshold

    # Dilate by 1 pixel
    binary = binary_dilation(binary, iterations=1)

    # Prune branches of length 5
    skeleton = thin(binary)
    skeleton = _prune_branches(skeleton, iterations=5)

    # Remove small components
    skeleton = remove_small_objects(skeleton, min_size=5)

    # Thin again
    skeleton = thin(skeleton)

    # Filter by size and frame span
    skeleton = _filter_components(skeleton, min_size, min_frames)

    # Thin once more
    skeleton = thin(skeleton)

    return skeleton


def analyze_wavelet_bidirectional(
    image_path: str | Path,
    threshold: float = 0.2,
    vision_net: torch.nn.Module | None = None,
    vision_threshold: float = 0.5,
    min_size: int = 10,
    min_frames: int = 10,
    device: str = "cpu",
) -> list[Track]:
    """Analyze a kymograph using wavelet-based segmentation + standard tracking.

    This is the neural-net-free alternative. The segmentation uses wavelets instead
    of UNet, but the tracking pipeline is the same as bidirectional.

    Corresponds to AnalyseKymographBIwavelet in WaveletPackage.wl lines 361-427.

    Args:
        image_path: Path to kymograph image.
        threshold: Binarization threshold for wavelet output.
        vision_net: Optional VisionNet for tracking (can be None for simpler tracking).
        vision_threshold: Threshold for vision module.
        min_size: Minimum track size in pixels.
        min_frames: Minimum track duration in frames.
        device: Computation device.

    Returns:
        List of detected tracks.
    """
    preprocessed, raw, was_negated = load_and_preprocess(image_path)

    h, w = preprocessed.shape
    if h > 5000 or w > 5000:
        raise ValueError(f"Image too large ({h}x{w}). Maximum 5000x5000 for wavelet mode.")

    # Wavelet segmentation produces a skeleton directly
    skeleton = _wavelet_segmentation(preprocessed, threshold, min_size, min_frames)

    # Create a "fake" prediction map from the skeleton (binary 0/1)
    # so we can reuse the bidirectional tracking pipeline
    prediction = skeleton.astype(np.float32)

    return track_bidirectional(
        prediction=prediction,
        kym_preprocessed=preprocessed,
        was_negated=was_negated,
        threshold=0.5,  # skeleton is already binary, use high threshold
        vision_threshold=vision_threshold,
        vision_net=vision_net,
        min_size=min_size,
        min_frames=min_frames,
        device=device,
    )
