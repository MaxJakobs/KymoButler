"""Neural network segmentation pipeline.

Corresponds to UniKymoButlerSegment and BiKymoButlerSegment in KymoButler.wl.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from kymobutler.morphology import check_image_size
from kymobutler.preprocessing import load_and_preprocess_mma, resize_to_multiple_of_16


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
        prediction_map is the float32 foreground probability channel at the network's
        resolution (both dimensions rounded to a multiple of 16), exactly like Mathematica's
        ``pred``. It is NOT resized back to the image: BiKymoButlerTrack binarizes it at this
        resolution and then resizes the binary image (see ``process_segmentation_bi``).

    Raises:
        ValueError: for an image smaller than 9 x 9 px, which Mathematica cannot segment
            either (``check_image_size``).
    """
    preprocessed, raw, was_negated, _ = load_and_preprocess_mma(image_path)
    check_image_size(preprocessed.shape)

    # Resize for net compatibility (Lanczos, from the float64 kympreproc)
    resized = resize_to_multiple_of_16(preprocessed)

    # Run network
    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)
    net.eval()
    with torch.no_grad():
        pred = net(tensor)  # (1, 2, H, W)

    # Foreground probability channel, kept at network resolution
    pred_map = pred[0, 0].cpu().numpy().astype(np.float32)

    return was_negated, raw.astype(np.float32), preprocessed.astype(np.float32), pred_map


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
        pred_dict has 'ant' and 'ret' keys with float32 probability maps at the network's
        resolution (both dimensions rounded to a multiple of 16), exactly like Mathematica's
        ``out``. They are NOT resized back to the image: UniKymoButlerTrack binarizes them at
        this resolution and then resizes the binary images (see ``process_segmentation_uni``).

    Raises:
        ValueError: for an image smaller than 9 x 9 px, which Mathematica cannot segment
            either (``check_image_size``).
    """
    preprocessed, raw, was_negated, _ = load_and_preprocess_mma(image_path)
    check_image_size(preprocessed.shape)

    resized = resize_to_multiple_of_16(preprocessed)

    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)
    net.eval()
    with torch.no_grad():
        pred = net(tensor)  # dict with 'ant', 'ret' each (1, 2, H, W)

    pred_dict = {key: pred[key][0, 0].cpu().numpy().astype(np.float32) for key in ("ant", "ret")}

    return was_negated, raw.astype(np.float32), preprocessed.astype(np.float32), pred_dict
