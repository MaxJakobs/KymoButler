"""Neural network segmentation pipeline.

Corresponds to UniKymoButlerSegment and BiKymoButlerSegment in KymoButler.wl.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from skimage.transform import resize

from kymobutler.preprocessing import load_and_preprocess, resize_to_multiple_of_16


def segment_bidirectional(
    image_path: str | Path,
    net: torch.nn.Module,
    device: str = "cpu",
) -> tuple[bool, np.ndarray, np.ndarray, np.ndarray]:
    """Segment a bidirectional kymograph using the UNET.

    Corresponds to BiKymoButlerSegment in KymoButler.wl lines 414-424.

    Args:
        image_path: Path to kymograph image.
        net: Loaded bidirectional segmentation UNet.
        device: Computation device.

    Returns:
        (was_negated, raw_grayscale, preprocessed, prediction_map)
        prediction_map is (H, W) float32, the foreground probability channel.
    """
    preprocessed, raw, was_negated = load_and_preprocess(image_path)
    original_dims = preprocessed.shape

    # Resize for net compatibility
    resized = resize_to_multiple_of_16(preprocessed)

    # Run network
    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)
    net.eval()
    with torch.no_grad():
        pred = net(tensor)  # (1, 2, H, W)

    # Extract foreground channel and resize back
    pred_map = pred[0, 0].cpu().numpy()  # foreground probability
    pred_map = resize(pred_map, original_dims, anti_aliasing=True, preserve_range=True).astype(
        np.float32
    )

    return was_negated, raw, preprocessed, pred_map


def segment_unidirectional(
    image_path: str | Path,
    net: torch.nn.Module,
    device: str = "cpu",
) -> tuple[bool, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Segment a unidirectional kymograph using the two-headed UNet.

    Corresponds to UniKymoButlerSegment in KymoButler.wl lines 46-58.

    Args:
        image_path: Path to kymograph image.
        net: Loaded unidirectional segmentation UNet.
        device: Computation device.

    Returns:
        (was_negated, raw_grayscale, preprocessed, pred_dict)
        pred_dict has 'ant' and 'ret' keys with (H, W) float32 probability maps.
    """
    preprocessed, raw, was_negated = load_and_preprocess(image_path)
    original_dims = preprocessed.shape

    resized = resize_to_multiple_of_16(preprocessed)

    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)
    net.eval()
    with torch.no_grad():
        pred = net(tensor)  # dict with 'ant', 'ret' each (1, 2, H, W)

    pred_dict = {}
    for key in ("ant", "ret"):
        p = pred[key][0, 0].cpu().numpy()
        pred_dict[key] = resize(p, original_dims, anti_aliasing=True, preserve_range=True).astype(
            np.float32
        )

    return was_negated, raw, preprocessed, pred_dict
