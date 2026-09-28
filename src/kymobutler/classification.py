"""Uni- vs bidirectional classification, as in the legacy KymoButler web service.

KymoButler.wl has no classification step: the free web form (DeployFreeWebform.wls) decided
the mode before calling UniKymoButler / BiKymoButler::

    rawkym = ImageAdjust@ColorConvert[ImageAdjust@RemoveAlphaChannel@kym, "Grayscale"]
    class  = classnet[ImageResize[rawkym, {64, 64}]]      (* 1 -> unidirectional *)

The classifier sees ``rawkym`` itself: not negated and not row-normalised (the training data,
KymoButler_ClassifierTraining.wls, were ImageAdjust'ed crops with random polarity). Because
``rawkym`` keeps the imported pixel type, ``ImageResize`` resamples it along x and then along
y, giving each pass that type: rounded to the 1/255 (1/65535) grid for an 8-bit (16-bit)
image, and for a 1-bit image, whose rawkym is Byte; rounded to float32 for a Real32 (32-bit
float or integer TIFF) image. Images with no side over 24 px are resized with a box kernel
(nearest neighbour), others with Lanczos, from 1 px wide (see
``kymobutler.preprocessing.image_resize``). The net has no mean/variance normalisation and its
classes are {1, 2} = {uni, bi}.

On 12 real kymographs this reproduces Mathematica's 64x64 input exactly and its
probabilities to 1e-6; on 408 random crops (both polarities) of the same images the decision
agrees in all 408 cases. The 64x64 input is also exact on the synthetic files of every pixel
format in tests/test_formats.py and tests/test_formats_extra.py (1 x 1 px and up, 1-bit,
8/16-bit, signed and 32-bit integer, float, palette, CMYK, rotated). ``image_resize`` is bit for
bit Mathematica's on every probed resize KymoButler can make: 545 Lanczos resizes (4.5 M
pixels of Byte/Bit16/Real32 images, Real64 within 1e-13) and every box-kernel resize to 64 px,
whose scale is a binary fraction (see ``kymobutler.preprocessing.box_taps`` for the ties of
other scales, which never occur here).

Public API (what the web app calls): ``classify_direction(image_path, classnet, device="cpu")``
returns ``"unidirectional"`` or ``"bidirectional"``; ``classify_probabilities`` of
``classifier_input(image_path)`` gives ``[p_uni, p_bi]``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from kymobutler.preprocessing import image_resize, import_image, mma_rawkym, rawkym_levels

CLASSIFIER_SIZE = (64, 64)


def classifier_input_from_pixels(
    pixels: np.ndarray, levels: int | None, *, real32: bool = False
) -> np.ndarray:
    """The classifier's 64x64 float32 input from ``import_image`` output (a crop of it is fine):
    ``ImageData@ImageResize[rawkym, {64, 64}]``.

    Args:
        pixels, levels: as returned by ``kymobutler.preprocessing.import_image`` /
            ``read_image`` (any size from 1 x 1 px).
        real32: the image is "Real32" (``import_image``'s image type), not "Real64".
    """
    raw = mma_rawkym(pixels, levels, real32=real32)
    return image_resize(
        raw, CLASSIFIER_SIZE, levels=rawkym_levels(levels), real32=real32
    ).astype(np.float32)


def classifier_input(image_path: str | Path) -> np.ndarray:
    """The classifier's 64x64 float32 input for a kymograph file."""
    pixels, levels, image_type = import_image(image_path)
    return classifier_input_from_pixels(pixels, levels, real32=image_type == "Real32")


def classify_probabilities(
    x64: np.ndarray, classnet: torch.nn.Module, device: str = "cpu"
) -> np.ndarray:
    """``[p_unidirectional, p_bidirectional]`` of a 64x64 classifier input."""
    x64 = np.ascontiguousarray(x64, dtype=np.float32)
    if x64.shape != CLASSIFIER_SIZE:
        raise ValueError(f"classifier input must be {CLASSIFIER_SIZE}, got {x64.shape}")
    tensor = torch.from_numpy(x64)[None, None].to(device)
    classnet.eval()
    with torch.no_grad():
        out = classnet(tensor)
    p = np.asarray(out.detach().cpu(), dtype=np.float64).ravel()
    if p.shape != (2,):
        raise ValueError(f"classifier returned {p.shape[0]} values, expected 2")
    return p


def classify_direction(
    image_path: str | Path, classnet: torch.nn.Module, device: str = "cpu"
) -> str:
    """``"unidirectional"`` or ``"bidirectional"`` for a kymograph file, as the legacy web form
    decided it. NetDecoder "Class" takes the most probable class, the first (uni) on a tie.

    Args:
        image_path: the uploaded file itself (any format ``import_image`` reads; never a
            re-encoded copy).
        classnet: the classifier, ``load_default_models(...)["classnet"]``.
        device: torch device.

    Raises:
        kymobutler.preprocessing.UnsupportedImageFormat: (a NotImplementedError) for files
            Mathematica cannot import or whose import is not reproduced, with a user-facing
            message starting ``"<file name>: "``.
        PIL.UnidentifiedImageError: if the file is not an image.
    """
    p = classify_probabilities(classifier_input(image_path), classnet, device)
    return "unidirectional" if p[0] >= p[1] else "bidirectional"
