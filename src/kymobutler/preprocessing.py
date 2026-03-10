"""Image preprocessing for kymograph analysis.

Corresponds to KymoButler.wl lines 39-55 (isNegated, normlines, UniKymoButlerSegment preprocessing).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from skimage.exposure import rescale_intensity
from skimage.transform import resize


def is_negated(img: np.ndarray) -> bool:
    """Detect if the kymograph has a white background (needs inversion).

    Binarizes at 0.5 and compares foreground pixel counts of original vs inverted.
    Corresponds to isNegated in KymoButler.wl line 39.
    """
    n1 = np.sum(img > 0.5)
    n2 = np.sum((1.0 - img) > 0.5)
    return bool(n1 >= n2)


def normalize_lines(img: np.ndarray) -> np.ndarray:
    """Normalize each row of the kymograph by its mean intensity.

    Rows with mean=0 are left unchanged.
    Corresponds to normlines in KymoButler.wl line 44.
    """
    means = img.mean(axis=1, keepdims=True)
    means = np.where(means > 0, means, 1.0)
    normalized = img / means
    return rescale_intensity(normalized.astype(np.float64), out_range=(0.0, 1.0)).astype(
        np.float32
    )


def resize_to_multiple_of_16(img: np.ndarray) -> np.ndarray:
    """Resize image so both dimensions are multiples of 16 (required by 4-level UNet).

    Corresponds to 16*Round@N[dim/16] in KymoButler.wl line 55.
    """
    h, w = img.shape[:2]
    new_h = max(16, 16 * round(h / 16))
    new_w = max(16, 16 * round(w / 16))
    if new_h == h and new_w == w:
        return img
    return resize(img, (new_h, new_w), anti_aliasing=True, preserve_range=True).astype(
        np.float32
    )


def load_and_preprocess(image_path: str | Path) -> tuple[np.ndarray, np.ndarray, bool]:
    """Full preprocessing pipeline for a kymograph image.

    Steps:
    1. Load image
    2. Remove alpha channel if present
    3. Convert to grayscale float32 [0, 1]
    4. Adjust intensity (rescale to full range)
    5. Detect polarity (white background -> invert)
    6. Normalize intensity per row

    Args:
        image_path: Path to the kymograph image.

    Returns:
        Tuple of (preprocessed_image, raw_grayscale, was_negated).
        - preprocessed_image: normalized grayscale float32 HxW
        - raw_grayscale: original grayscale float32 HxW (before negation/normalization)
        - was_negated: True if background was white and image was inverted
    """
    img = Image.open(image_path)

    # Remove alpha channel if present
    if img.mode == "RGBA":
        # Composite onto white background (matching Mathematica's RemoveAlphaChannel)
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, img)

    # Convert to grayscale
    img = img.convert("L")

    # To float32 [0, 1]
    raw = np.array(img, dtype=np.float32) / 255.0

    # ImageAdjust: rescale to full range
    raw = rescale_intensity(raw.astype(np.float64), out_range=(0.0, 1.0)).astype(np.float32)

    # Detect and correct polarity
    negated = is_negated(raw)
    preprocessed = 1.0 - raw if negated else raw.copy()

    # Normalize per line
    preprocessed = normalize_lines(preprocessed)

    return preprocessed, raw, negated
