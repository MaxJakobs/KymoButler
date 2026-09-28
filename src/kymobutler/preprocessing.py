"""Image preprocessing for kymograph analysis, reproducing Mathematica's image arithmetic.

Corresponds to KymoButler.wl lines 39-55 (isNegated, normlines and the preprocessing shared by
UniKymoButlerSegment and BiKymoButlerSegment)::

    rawkym = ImageAdjust@ColorConvert[ImageAdjust@RemoveAlphaChannel@kym, "Grayscale"]
    negkym = normlines@If[isNegated@rawkym, ColorNegate@rawkym, rawkym]
    net input = ImageData@ImageResize[negkym, 16*Round@N[dim/16]]

Mathematica keeps the pixel type of the imported image through ``rawkym``: a "Byte" (8-bit)
image is rounded to the 1/255 grid after every step, a "Bit16" one to the 1/65535 grid and a
"Real32" one to float32, while a "Real64" one is computed in double precision. A "Bit" (1-bit)
image becomes Byte in ``ColorConvert``. ``normlines`` makes a new ``Image`` of the row-normalised
data, which is Real32: ``negkym`` and the net input are float32 values.

Every rule below was established by probing Mathematica 14.3 (MacOSX-ARM64, default
ParallelThreadNumber 4) and is exact on the probes quoted in the docstrings. Pixel values are
passed around as float64 arrays (integer types scaled to [0, 1]) together with the pixel type:
``levels`` = 1 / 255 / 65535 for Bit / Byte / Bit16 and None for Real32 / Real64, and a
``real32`` flag telling Real32 from Real64 (see ``import_image``).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from PIL import Image, UnidentifiedImageError

_F32 = np.float32

# ColorConvert[., "Grayscale"] weights (Rec. 601 luma)
_GRAY_WEIGHTS = (0.299, 0.587, 0.114)

#: Mathematica's pixel types and the ``levels`` that stand for them
IMAGE_TYPE_LEVELS = {"Bit": 1, "Byte": 255, "Bit16": 65535, "Real32": None, "Real64": None}


class UnsupportedImageFormat(NotImplementedError):
    """An image file KymoButler cannot analyse the way Mathematica's KymoButler did.

    Raised for files that Mathematica's ``Import`` itself rejects or misreads (multi-frame files,
    1-bit TIFFs without a BitsPerSample tag, truncated TIFFs and TIFFs whose header declares no
    pixels, half-precision floats) and for pixel formats whose import is not reproduced here.
    The message starts with ``"<file name>: "`` and is meant for the user.
    """


# ------------------------------------------------------------------------------------ import
def import_image(image_path: str | Path) -> tuple[np.ndarray, int | None, str]:
    """``RemoveAlphaChannel@Import[file]``: pixel values, ``levels`` and Mathematica's image type.

    Returns:
        (pixels, levels, image_type): float64 ``(H, W)`` or ``(H, W, 3)`` array of the pixel
        values as Mathematica sees them; ``levels`` = 255 for "Byte", 65535 for "Bit16", 1 for
        "Bit" (values 0/1) and None for "Real32"/"Real64" (values unscaled; Real32 values are
        float32 numbers); ``image_type`` is Mathematica's ``ImageType`` of the import. Integer
        types are scaled to [0, 1] by ``levels``.

    Mathematica 14.3's import, reproduced here (probed on synthetic files of every kind):

    - 8-bit grey/RGB/palette PNG, TIFF, GIF, JPEG, WebP, BMP: Byte. Palette images are RGB, and
      so are 8-bit grey BMPs; a palette TIFF is the Bit16 RGB image of its 16-bit ColorMap. 1-bit
      PNG, BMP and TIFF: Bit. 2- and 4-bit grey PNGs are Byte with the stored values unscaled
      (``v/255``, v = 0..3 or 0..15); 2- and 4-bit grey TIFFs are Byte scaled to 0..255
      (``v*85``, ``v*17``). Of an MPO (a JPEG holding several images) the first image.
    - 16-bit grey, RGB, RGBA and grey+alpha, PNG and TIFF: Bit16 (Pillow alone would silently
      keep only the high bytes of 16-bit colour; they are read at full depth).
    - Signed integer TIFFs are offset: int8 -> Byte ``(v + 128)/255``, int16 -> Bit16
      ``(v + 32768)/65535`` (grey and RGB). 32-bit integer TIFFs become Real32:
      ``float32(v/(2^31 - 1))`` (int32) and ``float32(v/(2^32 - 1))`` (uint32). Float TIFFs:
      Real32/Real64, unscaled. Min-is-white TIFFs are negated (``_min_is_white``). BigTIFFs are
      read like classic TIFFs, except that their Orientation tag is ignored.
    - CMYK (JPEG and 8/16-bit TIFF): returned as the RGB image Mathematica's ``ImageAdjust``
      converts it to (``_cmyk_to_rgb``), which is what every KymoButler step sees
      (``RemoveAlphaChannel`` keeps CMYK). Adobe CMYK JPEGs are read without undoing the Adobe
      inversion, as Mathematica does.
    - Compressed TIFFs (LZW, Deflate, PackBits, ...) via tifffile + imagecodecs. TIFFs whose
      compression cannot be decoded raise.
    - Orientation: the Exif Orientation of a JPEG, MPO or WebP and a (classic) TIFF's
      Orientation tag are applied as Mathematica applies them (``_MMA_ORIENTATION``: 2 flips top
      to bottom and 4 left to right, the reverse of the Exif/TIFF convention; 3 and 5-8 as the
      convention says; 0, 9 and other values are ignored). An orientation given only in XMP
      metadata and a PNG's Exif orientation are ignored. The pixels are never turned the way
      Pillow would turn them (``_read_pil_tiff_as_stored``).
    - ``RemoveAlphaChannel`` drops the alpha channel (it does not composite onto white). A TIFF
      extra sample is an alpha channel to Mathematica whatever its ExtraSamples type.

    Raises:
        UnsupportedImageFormat: for a multi-frame file (Mathematica's ``Import`` returns a list
            of images, on which KymoButler fails), a 1-bit TIFF without a BitsPerSample tag
            (the tag is optional for 1-bit TIFFs and Pillow and tifffile leave it out of
            uncompressed ones; Mathematica's ``Import`` fails on it), a TIFF whose header
            declares a height or width of 0 (Import fails), a half-precision float TIFF
            (Mathematica reads the bits as signed 16-bit integers, (bits XOR 0x8000)/65535,
            which scrambles the intensities), a TIFF without any image (e.g. truncated; Import
            fails), and pixel formats whose import is not reproduced (e.g. YCbCr/Lab TIFFs,
            64-bit integers, palette TIFFs with an alpha channel).
        PIL.UnidentifiedImageError: for a file that is not an image.
    """
    path = Path(image_path)
    try:
        img = Image.open(path)
    except UnidentifiedImageError:
        if _is_tiff(path):  # e.g. float64, 16-bit grey+alpha or planar 16-bit colour TIFFs
            return _checked(_read_tiff(path), path)
        raise
    with img:
        n_frames = getattr(img, "n_frames", 1)
        if n_frames > 1 and img.format != "MPO":  # of an MPO, Mathematica imports the 1st image
            raise UnsupportedImageFormat(
                f"{path.name}: the file holds {n_frames} images (frames/pages). Mathematica's "
                "Import returns a list of images for it, which KymoButler cannot analyse; save "
                "the kymograph as a single-image file"
            )
        if img.format == "TIFF":
            if 258 not in img.tag_v2:
                raise UnsupportedImageFormat(
                    f"{path.name}: Mathematica cannot import this 1-bit TIFF (it has no "
                    "BitsPerSample tag); save the kymograph as an 8-bit grayscale image"
                )
            if _tiff_needs_tifffile(img):
                return _checked(_read_tiff(path), path)
            if (_orientation_value(img.tag_v2.get(274)) in _MMA_ORIENTATION
                    or _pillow_orientation(img) in _MMA_ORIENTATION):  # Pillow may turn it
                return _checked(_read_pil_tiff_as_stored(path), path)
            return _checked(_read_pil(path, img), path)
        if img.format == "PNG" and img.mode in ("RGB", "RGBA") and _is_16bit(img):
            return _checked(_read_png_16bit(path, img), path)
        values, image_type = _read_pil(path, img)
        if img.format in _EXIF_ORIENTED_FORMATS:
            values = _apply_orientation(values, _exif_orientation(img))
        return _checked((values, image_type), path)


def read_image(image_path: str | Path) -> tuple[np.ndarray, int | None]:
    """``RemoveAlphaChannel@Import[file]`` as ``(pixels, levels)``; see ``import_image`` (which
    also tells Real32 from Real64 images)."""
    pixels, levels, _ = import_image(image_path)
    return pixels, levels


def _checked(read: tuple[np.ndarray, str], path: Path) -> tuple[np.ndarray, int | None, str]:
    values, image_type = read
    if values.size == 0:
        raise UnsupportedImageFormat(
            f"{path.name}: the file describes an image of {values.shape[0]} x "
            f"{values.shape[1] if values.ndim > 1 else 0} px (height x width), i.e. without "
            "pixels (Mathematica's Import fails on it); open it in Fiji/ImageJ and save it again"
        )
    return values, IMAGE_TYPE_LEVELS[image_type], image_type


# Orientation value (JPEG Exif, TIFF tag 274) -> the transform Mathematica 14.3's Import applies
# to the stored pixels. Probed on every value 0..9 on JPEGs and on TIFFs of every reader path
# (8/16-bit grey, RGB, 16-bit RGB, float, 1-bit, uncompressed and LZW). 2 and 4 are swapped
# relative to the Exif/TIFF convention (where 2 mirrors left to right); 0, 9 and other values
# leave the image as stored.
_MMA_ORIENTATION = {
    2: np.flipud,
    3: lambda a: np.rot90(a, 2),
    4: np.fliplr,
    5: lambda a: np.swapaxes(a, 0, 1),
    6: lambda a: np.rot90(a, -1),
    7: lambda a: np.swapaxes(np.rot90(a, 2), 0, 1),
    8: lambda a: np.rot90(a, 1),
}


def _orientation_value(v) -> int | None:
    try:
        return int(v[0] if isinstance(v, (tuple, list)) else v)
    except (TypeError, ValueError, IndexError):
        return None  # not a number: no orientation (ignored like 0 or 9)


def _apply_orientation(values: np.ndarray, orientation) -> np.ndarray:
    f = _MMA_ORIENTATION.get(_orientation_value(orientation))
    return values if f is None else np.ascontiguousarray(f(values))


# Formats whose Exif Orientation Mathematica 14.3's Import applies (probed: JPEG, the first
# image of an MPO, WebP); it ignores a PNG's (eXIf chunk) and every XMP orientation.
_EXIF_ORIENTED_FORMATS = ("JPEG", "MPO", "WEBP")


def _exif_orientation(img: Image.Image) -> int | None:
    """The Orientation of the image's Exif block. Unlike ``Image.getexif``, an orientation given
    only in the XMP metadata is not used: Mathematica ignores it (probed)."""
    raw = img.info.get("exif")
    if not raw:
        return None
    exif = Image.Exif()
    exif.load(raw)
    return _orientation_value(exif.get(274))


def _pillow_orientation(img: Image.Image) -> int | None:
    """The orientation Pillow applies to a TIFF when it loads it (Pillow's TIFF reader turns the
    pixels by its ``exif_transpose``, which also takes an XMP orientation when there is no
    Orientation tag)."""
    return _orientation_value(img.getexif().get(274))


def _read_pil_tiff_as_stored(path: Path) -> tuple[np.ndarray, str]:
    """A TIFF that Pillow would turn on loading (``_pillow_orientation``), read by Pillow as
    stored and then turned as Mathematica turns it.

    The Orientation tag (and an XMP orientation) of an in-memory copy of the file are set to 1,
    so Pillow reads the stored pixels exactly as it reads an unturned TIFF; Mathematica's
    transform for the file's own Orientation tag is applied to them (an XMP orientation is
    ignored, as Mathematica ignores it).
    """
    import io

    data, orientation = _tiff_without_orientation(path)
    if _is_bigtiff(path):
        orientation = None  # Mathematica ignores the Orientation tag of a BigTIFF (probed)
    with Image.open(io.BytesIO(data)) as img:
        if (_orientation_value(img.tag_v2.get(274)) in _MMA_ORIENTATION
                or _pillow_orientation(img) in _MMA_ORIENTATION):
            raise RuntimeError(f"{path.name}: could not read the TIFF without Pillow turning it")
        values, image_type = _read_pil(path, img)
    return _apply_orientation(values, orientation), image_type


_XMP_ORIENTATION = re.compile(rb'tiff:Orientation(="|>)([0-9])')  # Pillow's pattern
_TIFF_INT_SIZES = {1: 1, 3: 2, 4: 4, 6: 1, 8: 2, 9: 4, 16: 8, 17: 8}  # integer DATATYPEs


def _tiff_without_orientation(path: Path) -> tuple[bytes, int | None]:
    """The file's bytes with the first page's Orientation tag and XMP orientation set to 1, and
    the Orientation tag's value (None without one)."""
    import tifffile

    data = bytearray(path.read_bytes())
    with tifffile.TiffFile(str(path)) as tf:
        page = tf.pages[0]
        order = "little" if tf.byteorder == "<" else "big"
        tag, xmp = page.tags.get(274), page.tags.get(700)
        orientation = None
        if tag is not None:
            size = _TIFF_INT_SIZES.get(int(tag.dtype))
            if size is None or tag.count != 1:
                raise UnsupportedImageFormat(
                    f"{path.name}: the TIFF's Orientation tag is malformed (type "
                    f"{int(tag.dtype)}, {tag.count} values); save the kymograph again"
                )
            off = tag.valueoffset
            orientation = int.from_bytes(data[off:off + size], order, signed=False)
            data[off:off + size] = (1).to_bytes(size, order)
        if xmp is not None:
            lo = xmp.valueoffset
            for m in _XMP_ORIENTATION.finditer(bytes(data[lo:lo + xmp.count])):
                data[lo + m.start(2)] = ord("1")
    return bytes(data), orientation


def _is_tiff(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(4) in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def _is_bigtiff(path: Path) -> bool:
    with open(path, "rb") as f:
        return f.read(4) in (b"II+\x00", b"MM\x00+")


def _rawmode(img: Image.Image) -> str:
    args = img.tile[0][3] if img.tile else ""
    return str(args if isinstance(args, str) else (args[0] if args else ""))


def _is_16bit(img: Image.Image) -> bool:
    """Whether a PIL image opened in an 8-bit mode really has 16-bit samples."""
    if any(";16" in str(t[3] if isinstance(t[3], str) else (t[3][0] if t[3] else ""))
           for t in img.tile):
        return True
    if img.format == "TIFF":
        bits = img.tag_v2.get(258, (8,))
        return max(bits if isinstance(bits, (tuple, list)) else (int(bits),)) > 8
    return False


def _tiff_tag(img: Image.Image, tag: int, default: int) -> int:
    v = img.tag_v2.get(tag, default)
    return int(v[0] if isinstance(v, (tuple, list)) else v)


def _tiff_needs_tifffile(img: Image.Image) -> bool:
    """TIFFs whose samples Pillow does not return as stored: 16-bit colour / grey+alpha (Pillow
    keeps the high bytes), RGB with associated alpha (Pillow un-premultiplies) and RGB with an
    unspecified extra sample."""
    if img.mode in ("RGB", "RGBA", "LA", "CMYK") and _is_16bit(img):
        return True
    return img.format == "TIFF" and _rawmode(img) in ("RGBa", "RGBX")


def _pil_inverts(rawmode: str) -> bool:
    """Whether Pillow unpacks this raw mode inverted (min-is-white 1- and 8-bit TIFFs)."""
    return any(t in ("I", "IR") for t in rawmode.split(";")[1:])


def _read_pil(path: Path, img: Image.Image) -> tuple[np.ndarray, str]:
    rawmode = _rawmode(img)  # before loading: Pillow clears the tile list on load
    values, image_type = _read_pil_samples(path, img, rawmode)
    if img.format == "TIFF" and _tiff_tag(img, 262, 1) == 0 and not _pil_inverts(rawmode):
        values = _min_is_white(values, image_type)  # Pillow inverts only 1- and 8-bit ones
    return values, image_type


def _read_pil_samples(path: Path, img: Image.Image, rawmode: str) -> tuple[np.ndarray, str]:
    mode = img.mode
    name = path.name
    if img.format == "TIFF" and mode == "P":
        return _palette_rgb(np.asarray(img), img.tag_v2.get(320), name)
    if img.format == "TIFF" and mode == "PA":
        raise UnsupportedImageFormat(
            f"{name}: palette TIFFs with an alpha channel are not supported (Mathematica's "
            "import of them is not reproduced); save the kymograph as 8-bit or 16-bit grayscale"
        )
    if img.format == "BMP" and mode in ("L", "P"):
        # Mathematica imports 8-bit (grey or palette) BMPs as RGB images
        return np.asarray(img.convert("RGB")) / 255.0, "Byte"
    if mode == "1":
        return np.asarray(img, dtype=np.float64), "Bit"
    if mode == "L":
        arr = np.asarray(img).astype(np.int64)
        if img.format == "TIFF" and _tiff_tag(img, 339, 1) == 2:  # signed int8: offset binary
            arr = (arr + 128) % 256
        if img.format == "PNG" and rawmode in ("L;2", "L;4"):
            # Pillow scales 2/4-bit grey to 0..255; Mathematica keeps the stored values
            arr //= 85 if rawmode == "L;2" else 17
        return arr / 255.0, "Byte"
    if mode == "LA":
        return np.asarray(img)[..., 0] / 255.0, "Byte"
    if mode in ("P", "PA"):
        return np.asarray(img.convert("RGBA"))[..., :3] / 255.0, "Byte"
    if mode in ("RGB", "RGBA"):
        return np.asarray(img)[..., :3] / 255.0, "Byte"
    if mode in ("I;16", "I;16B", "I;16L", "I;16N"):
        return np.asarray(img).astype(np.float64) / 65535.0, "Bit16"
    if mode == "I":
        arr = np.asarray(img).astype(np.int64)
        if img.format == "PNG":  # Pillow < 10 opens 16-bit grey PNGs as "I"
            return arr / 65535.0, "Bit16"
        if img.format == "TIFF":
            bits, fmt = _tiff_tag(img, 258, 1), _tiff_tag(img, 339, 1)
            if (bits, fmt) == (16, 2):
                return _mma_samples(arr.astype(np.int16), name)
            if (bits, fmt) == (32, 2):
                return _mma_samples(arr.astype(np.int32), name)
            if (bits, fmt) == (32, 1):
                return _mma_samples((arr % 2**32).astype(np.uint32), name)
    if mode == "F" and img.format == "TIFF" and _tiff_tag(img, 258, 32) == 32:
        return np.asarray(img).astype(np.float64), "Real32"
    if mode == "CMYK" and not _is_16bit(img):
        arr = np.asarray(img)
        if img.format in ("JPEG", "MPO") and rawmode == "CMYK;I":
            arr = 255 - arr  # Mathematica keeps the Adobe-inverted samples
        return _cmyk_to_rgb(arr / 255.0, 255), "Byte"
    raise UnsupportedImageFormat(
        f"{name}: images of this pixel format ({img.format} {mode}) are not supported "
        "(Mathematica's import of it is not reproduced); save the kymograph as 8-bit or 16-bit "
        "grayscale, 8-bit or 16-bit RGB or 32-bit float"
    )


def _min_is_white(values: np.ndarray, image_type: str) -> np.ndarray:
    """A min-is-white TIFF (PhotometricInterpretation 0): Mathematica 14.3 imports the samples
    as for min-is-black and negates them, ``1 - x`` on the type's grid, in float32 for Real32
    (probed on 1-, 8- and 16-bit, signed 16-bit and 32-bit float TIFFs, with and without an
    alpha channel)."""
    levels = IMAGE_TYPE_LEVELS[image_type]
    if levels is not None:
        return (levels - np.rint(values * levels)) / levels
    if image_type == "Real32":
        return (_F32(1) - values.astype(_F32)).astype(np.float64)
    return 1.0 - values


def _palette_rgb(index: np.ndarray, colormap, name: str) -> tuple[np.ndarray, str]:
    """A palette TIFF: Mathematica 14.3 imports it as the Bit16 RGB image of its 16-bit
    ColorMap, ``colormap/65535`` (probed; Pillow would round the colours to 8 bits)."""
    if colormap is None:
        raise UnsupportedImageFormat(f"{name}: this palette TIFF has no ColorMap tag")
    cmap = np.asarray(colormap, dtype=np.int64).reshape(3, -1)
    index = np.asarray(index, dtype=np.int64)
    if index.ndim != 2 or index.size and index.max() >= cmap.shape[1]:
        raise UnsupportedImageFormat(f"{name}: this palette TIFF's pixels and ColorMap disagree")
    return np.moveaxis(cmap[:, index], 0, -1) / 65535.0, "Bit16"


def _cmyk_to_rgb(cmyk: np.ndarray, levels: int) -> np.ndarray:
    """The RGB image Mathematica 14.3's ``ImageAdjust`` makes of a Byte/Bit16 CMYK image before
    adjusting it: ``R = 1 - fma(C, 1 - K, K)`` (G, B alike) in single precision, on the channel
    values ``k/levels`` as float32, then ``floor(R*levels + 1/2)`` in float32.

    Exact on all 71,040 pixels of a 16-bit probe whose ``(1 - C)(1 - K)`` lie within 0.004
    levels of a rounding tie (double-precision ``(1 - C)(1 - K)`` rounded gets 11,556 of them
    wrong) and on random 8/16-bit images; for 8-bit images every (C, K) pair gives the same as
    the double-precision product (8-bit products are never that close to a tie).
    """
    q = np.rint(np.asarray(cmyk, dtype=np.float64) * levels).astype(_F32) / _F32(levels)
    c = q[..., :3]
    k = np.broadcast_to(q[..., 3:4], c.shape)
    y = _F32(1) - _fma32(c, _F32(1) - k, k)
    return np.floor(y * _F32(levels) + _F32(0.5)).astype(np.float64) / levels


# Mathematica's Import of integer/float TIFF samples
def _mma_samples(arr: np.ndarray, name: str) -> tuple[np.ndarray, str]:
    dt = arr.dtype
    if dt == np.bool_:
        return arr.astype(np.float64), "Bit"
    if dt == np.uint8:
        return arr / 255.0, "Byte"
    if dt == np.int8:
        return (arr.astype(np.int64) + 128) / 255.0, "Byte"
    if dt == np.uint16:
        return arr / 65535.0, "Bit16"
    if dt == np.int16:
        return (arr.astype(np.int64) + 32768) / 65535.0, "Bit16"
    if dt == np.int32:
        return (arr.astype(np.float64) / 2147483647.0).astype(_F32).astype(np.float64), "Real32"
    if dt == np.uint32:
        return (arr.astype(np.float64) / 4294967295.0).astype(_F32).astype(np.float64), "Real32"
    if dt == np.float32:
        return arr.astype(np.float64), "Real32"
    if dt == np.float64:
        return arr.astype(np.float64), "Real64"
    if dt == np.float16:
        raise UnsupportedImageFormat(
            f"{name}: half-precision (16-bit float) TIFFs are not supported: Mathematica misreads "
            "them as 16-bit integers; save the kymograph as a 32-bit float or 16-bit integer TIFF"
        )
    raise UnsupportedImageFormat(
        f"{name}: TIFF samples of type {dt} are not supported (Mathematica's import of them is "
        "not reproduced); save the kymograph as an 8-bit, 16-bit or 32-bit float TIFF"
    )


def _read_tiff(path: Path) -> tuple[np.ndarray, str]:
    """A TIFF Pillow cannot read as stored, read with tifffile."""
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - tifffile is a dependency
        raise ImportError("reading this TIFF needs the 'tifffile' package") from exc
    name = path.name
    try:
        with tifffile.TiffFile(str(path)) as tf:
            n = len(tf.pages)
            if n == 0:
                raise UnsupportedImageFormat(
                    f"{name}: the file holds no image, so it is probably truncated or corrupt "
                    "(Mathematica's Import fails on it); save the kymograph again"
                )
            if n > 1:
                raise UnsupportedImageFormat(
                    f"{name}: the file holds {n} images (frames/pages). Mathematica's Import "
                    "returns a list of images for it, which KymoButler cannot analyse; save the "
                    "kymograph as a single-image file"
                )
            page = tf.pages[0]
            height, width = int(page.imagelength), int(page.imagewidth)
            if height == 0 or width == 0:
                raise UnsupportedImageFormat(
                    f"{name}: the TIFF header declares an image of {height} x {width} px "
                    "(height x width), i.e. without pixels, so the file is probably corrupt "
                    "(Mathematica's Import fails on it); open it in Fiji/ImageJ and save it again"
                )
            photometric = int(page.photometric)
            colormap = page.colormap if photometric == 3 else None
            axes = page.axes
            orientation = page.tags.get(274)
            if orientation is None or tf.is_bigtiff:  # Mathematica ignores a BigTIFF's
                orientation = None
            else:
                orientation = orientation.value
            arr = page.asarray()
    except UnsupportedImageFormat:
        raise
    except Exception as exc:  # e.g. corrupt data or a codec imagecodecs does not provide
        raise UnsupportedImageFormat(
            f"{name}: this TIFF could not be decoded ({exc}); save it uncompressed or with "
            "LZW/Deflate/PackBits compression"
        ) from None
    if axes == "SYX":
        arr = np.moveaxis(arr, 0, -1)
    elif axes not in ("YX", "YXS"):
        raise UnsupportedImageFormat(f"{name}: unsupported TIFF layout {axes} {arr.shape}")
    if photometric == 3 and arr.ndim == 2:  # palette
        values, image_type = _palette_rgb(arr, colormap, name)
        return _apply_orientation(values, orientation), image_type
    values, image_type = _mma_samples(arr, name)
    n_ch = 1 if values.ndim == 2 else values.shape[-1]
    n_color = {0: 1, 1: 1, 2: 3, 5: 4}.get(photometric)  # MINISWHITE/BLACK, RGB, CMYK
    if n_color is None or n_ch - n_color not in (0, 1):
        raise UnsupportedImageFormat(
            f"{name}: TIFFs with photometric interpretation {photometric} and {n_ch} samples per "
            "pixel are not supported (Mathematica's import of them is not reproduced)"
        )
    if n_ch > n_color:  # RemoveAlphaChannel (any extra sample is alpha to Mathematica)
        values = values[..., :n_color]
    if n_color == 1 and values.ndim == 3:
        values = values[..., 0]
    if photometric == 0:
        values = _min_is_white(values, image_type)
    if photometric == 5:
        levels = IMAGE_TYPE_LEVELS[image_type]
        if levels is None or levels == 1:
            raise UnsupportedImageFormat(
                f"{name}: CMYK TIFFs of type {arr.dtype} are not supported"
            )
        values = _cmyk_to_rgb(values, levels)
    return _apply_orientation(values, orientation), image_type


def _read_png_16bit(path: Path, img: Image.Image) -> tuple[np.ndarray, str]:
    """A 16-bit RGB / RGBA / grey+alpha PNG at full depth.

    Pillow has no 16-bit colour modes and unpacks only the high byte of each sample. The
    decoder is run twice, once unpacking the high and once the low byte of every sample
    (the same filter/interlace handling, as both unpackers have the same bits per pixel); a
    grey+alpha PNG is unpacked as raw 4-byte pixels.
    """
    rawmode = _rawmode(img)
    if rawmode in ("RGB;16B", "RGBA;16B"):
        hi = _decode_png(path, rawmode)
        lo = _decode_png(path, rawmode.replace(";16B", ";16L"))
        if not np.array_equal(hi, np.asarray(img)):
            raise RuntimeError(f"{path.name}: unexpected 16-bit PNG decoding by Pillow")
        arr = (hi.astype(np.uint16) << 8) | lo.astype(np.uint16)
        return arr[..., :3] / 65535.0, "Bit16"
    if rawmode == "LA;16B":
        raw = _decode_png(path, "RGBA").astype(np.uint16)
        grey = (raw[..., 0] << 8) | raw[..., 1]
        if not np.array_equal(raw[..., 0], np.asarray(img)[..., 0]):
            raise RuntimeError(f"{path.name}: unexpected 16-bit PNG decoding by Pillow")
        return grey / 65535.0, "Bit16"
    raise UnsupportedImageFormat(f"{path.name}: unsupported 16-bit PNG layout {rawmode!r}")


def _decode_png(path: Path, rawmode: str) -> np.ndarray:
    with Image.open(path) as im:
        t = im.tile[0]
        im.tile = [type(t)(t[0], t[1], t[2], rawmode)] if hasattr(t, "_fields") else [
            (t[0], t[1], t[2], rawmode)
        ]
        im.load()
        return np.asarray(im)


# ------------------------------------------------------------------- floating-point helpers
# Mathematica's image kernels are compiled with fused multiply-adds (FMA) on this machine
# (MacOSX-ARM64); ColorConvert's weighted sums are only reproduced exactly with them.
def _two_sum(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _two_prod(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    def split(v):
        t = 134217729.0 * v
        hi = t - (t - v)
        return hi, v - hi

    p = a * b
    ah, al = split(a)
    bh, bl = split(b)
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _round_to_odd(s: np.ndarray, e: np.ndarray) -> np.ndarray:
    """The float64 of ``s + e`` rounded to odd, for ``s = RN(s + e)`` and the exact error ``e``."""
    fix = (e != 0) & ((s.view(np.int64) & 1) == 0)
    if not fix.any():
        return s
    return np.where(fix, np.nextafter(s, np.where(e > 0, np.inf, -np.inf)), s)


def _fma32(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """float32 ``a*b + c`` rounded once (a, b, c float32): ``a*b`` is exact in float64 and
    rounding the float64 sum to odd before rounding to float32 is correct rounding."""
    p = a.astype(np.float64) * b.astype(np.float64)
    s, e = _two_sum(p, c.astype(np.float64))
    return _round_to_odd(s, e).astype(_F32)


def _fma64(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """float64 ``a*b + c`` rounded once (Boldo & Melquiond's emulation with rounding to odd)."""
    uh, ul = _two_prod(a, b)
    th, tl = _two_sum(c, uh)
    return th + _round_to_odd(*_two_sum(tl, ul))


# ------------------------------------------------------------------------------ ImageAdjust
# Mathematica computes an image's min and max with a parallel reduction (ParallelThreadNumber 4
# by default on the reference machine) above VectorParallelLengthThresholds = 8192 elements.
# For a channel of n > 8192 pixels with n = 1 (mod 4), that reduction leaves out the image's
# last pixel (bottom right, row-major). Probed on Byte/Bit16/Real32/Real64 images, single- and
# multi-channel: every n in 4097..12289 (only n = 1 mod 4 above 8192 is affected), every
# position of a unique extreme in a 9049 px image (only the last is ignored), 2-D shapes up to
# 1001 x 1001, and 140 fresh random images of 9001..60501 px (49 of them with n = 1 mod 4 and a
# unique extreme at the end, where the quirk changes thousands of pixels). With
# ParallelThreadNumber -> 1 Mathematica has no such effect; with another number of threads the
# rule would presumably differ (not probed).
_PARALLEL_MINMAX_THRESHOLD = 8192
_PARALLEL_MINMAX_THREADS = 4


def _minmax_region(flat: np.ndarray) -> np.ndarray:
    """The pixels (rows of ``flat``, one per pixel) Mathematica's ImageAdjust takes the min and
    max over."""
    n = flat.shape[0]
    if n > _PARALLEL_MINMAX_THRESHOLD and n % _PARALLEL_MINMAX_THREADS == 1:
        return flat[:-1]
    return flat


def _adjust_levels(k: np.ndarray, lo: int, hi: int, levels: int) -> np.ndarray:
    """ImageAdjust of one channel of an integer image, given as integer levels ``k`` and the
    min/max levels Mathematica uses; returns the output levels (a lookup table per level)."""
    if lo == hi:
        return np.zeros_like(k)
    kmin, kmax = int(k.min()), int(k.max())
    u = np.arange(kmin, kmax + 1)
    x, a, b = u / levels, lo / levels, hi / levels
    y = ((x - a) / (b - a) - 0.5) + 0.5
    out = np.floor(np.clip(y, 0.0, 1.0) * levels + 0.5)
    return out.astype(np.int64)[k - kmin]


def image_adjust(x: np.ndarray, levels: int | None, *, real32: bool = False) -> np.ndarray:
    """``ImageAdjust[img]``: stretch every channel to [0, 1] (a constant channel becomes 0).

    Mathematica 14.3 computes ``((x - lo)/(hi - lo) - 1/2) + 1/2`` in double precision (the
    contrast step of ``ImageAdjust[img, {0, 0, 1}]``), clips to [0, 1] and rounds the result
    onto the image type: half up onto the 1/255 (1/65535) grid for Byte (Bit16), to float32 for
    Real32 (``real32=True``). ``lo``/``hi`` are the channel's min and max, except for
    Mathematica's parallel min/max quirk (``_minmax_region``). A Real32/Real64 channel whose
    ``lo``/``hi`` are exactly 0 and 1 is returned unchanged (the formula would move values below
    1/4 by up to 2^-54; for Byte/Bit16 the rounding hides the difference). Exact on every level
    of all 32,640 Byte ranges and of 451 Bit16 ranges (1.7 M levels) and on 1,260 random
    Byte/Bit16/Real32/Real64 grey and RGB images. A fresh probe set (540 images and 300 Bit16
    ranges) had 148,558 Byte and 1,648 Bit16 exact rounding ties, n + 1/2 levels: all match,
    where plain ``(x - lo)/(hi - lo)`` rounded half up gets 619 of them wrong; a second fresh
    set (Mathematica 14.3, 2026-09-28: 200 Byte images with 36,215 exact ties, 60 Real32 images,
    285 Bit16 ranges up to the full 0..65535 with 1,398 exact ties, 2.1 M levels) matches bit for
    bit too, as do 32 Real32/Real64 images around the [0, 1] rule (per channel; min -0.0 counts
    as 0; [0, 1 - 2^-53], [2^-60, 1], [0, 1 + 2^-52] and the min/max quirk's excluded pixel
    are not exempt).
    """
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(-1, x.shape[-1]) if x.ndim == 3 else x.reshape(-1, 1)
    region = _minmax_region(flat)
    out = np.empty_like(flat)
    for c in range(flat.shape[1]):
        v = flat[:, c]
        if levels is None:
            lo, hi = region[:, c].min(), region[:, c].max()
            if hi == lo:
                out[:, c] = 0.0
            elif lo == 0.0 and hi == 1.0:
                out[:, c] = v
            else:
                out[:, c] = np.clip(((v - lo) / (hi - lo) - 0.5) + 0.5, 0.0, 1.0)
            continue
        k = np.rint(v * levels).astype(np.int64)
        kr = k if region is flat else k[:-1]
        out[:, c] = _adjust_levels(k, int(kr.min()), int(kr.max()), levels) / levels
    if real32 and levels is None:
        out = out.astype(_F32).astype(np.float64)
    return out.reshape(x.shape)


def color_convert_grayscale(
    x: np.ndarray, levels: int | None, *, real32: bool = False
) -> np.ndarray:
    """``ColorConvert[rgbImage, "Grayscale"]``: 0.299 R + 0.587 G + 0.114 B.

    Mathematica evaluates ``fma(B, 0.114, fma(R, 0.299, G*0.587))`` (fused multiply-adds), in
    single precision for Byte, Bit16 and Real32 images (on the channel values ``k/levels`` as
    float32 for integer types, then ``floor(y*levels + 1/2)``, all in float32) and in double
    precision for Real64 images. Real32/Real64 channel values are first clipped to [0, 1]
    (8,000 probed colours out of range: exact). Exact on all 16,777,216 Byte colours, 600,000
    Bit16 colours (400,000 of them chosen near ties) and 200,000 random Real32 and Real64
    colours each. Grayscale input is returned unchanged (a Bit image becomes Byte, with the same
    0/1 values).
    """
    if x.ndim == 2:
        return x
    wr, wg, wb = _GRAY_WEIGHTS
    if levels is None:
        x = np.clip(np.asarray(x[..., :3], dtype=np.float64), 0.0, 1.0)
    if levels is None and not real32:  # Real64
        r, g, b = (x[..., i] for i in range(3))
        return _fma64(b, np.full_like(b, wb), _fma64(r, np.full_like(r, wr), g * wg))
    if levels is None:
        c = np.asarray(x, dtype=_F32)
    else:
        c = np.rint(np.asarray(x[..., :3]) * levels).astype(_F32) / _F32(levels)
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    y = _fma32(b, np.full_like(b, _F32(wb)), _fma32(r, np.full_like(r, _F32(wr)), g * _F32(wg)))
    if levels is None:
        return y.astype(np.float64)
    return np.floor(y * _F32(levels) + _F32(0.5)).astype(np.float64) / levels


def rawkym_levels(levels: int | None) -> int | None:
    """Pixel type of ``rawkym`` for an imported image of type ``levels``: a Bit image becomes
    Byte in ``ColorConvert[., "Grayscale"]`` (Mathematica 14.3), every other type is kept."""
    return 255 if levels == 1 else levels


def mma_rawkym(pixels: np.ndarray, levels: int | None, *, real32: bool = False) -> np.ndarray:
    """``rawkym = ImageAdjust@ColorConvert[ImageAdjust@pixels, "Grayscale"]`` (float64, on the
    grid of ``rawkym_levels(levels)``; float32 values for a Real32 image)."""
    x = image_adjust(pixels, levels, real32=real32)
    x = color_convert_grayscale(x, levels, real32=real32)
    return image_adjust(x, rawkym_levels(levels), real32=real32)


def _load_rawkym(image_path: str | Path) -> tuple[np.ndarray, int | None, bool]:
    pixels, levels, image_type = import_image(image_path)
    real32 = image_type == "Real32"
    return mma_rawkym(pixels, levels, real32=real32), rawkym_levels(levels), real32


def load_grayscale(image_path: str | Path) -> tuple[np.ndarray, int | None]:
    """``rawkym`` of an image file. Returns (rawkym float64 (H, W), levels of its pixel type,
    see ``rawkym_levels``)."""
    raw, levels, _ = _load_rawkym(image_path)
    return raw, levels


def is_negated(img: np.ndarray) -> bool:
    """Detect if the kymograph has a white background (needs inversion).

    ``isNegated`` (KymoButler.wl line 39): ``Total@Binarize[kym, .5] >=
    Total@Binarize[ColorNegate@kym, .5]``, Binarize keeping values strictly above 0.5.
    """
    n1 = np.sum(img > 0.5)
    n2 = np.sum((1.0 - img) > 0.5)
    return bool(n1 >= n2)


def _normlines(img: np.ndarray) -> np.ndarray:
    """``normlines`` (KymoButler.wl line 44):
    ``ImageAdjust@Image@Map[If[Mean[#] > 0., #/Mean@#, #] &, ImageData@kym]``.

    The rows are divided by their means in double precision; ``Image`` of that data is a Real32
    image, which ImageAdjust keeps: the result is float32 values (float64 array). Exact on the
    ``negkym`` of 12 real kymographs and 27 synthetic images (Byte, Bit16, Real32, Real64).
    """
    img = np.asarray(img, dtype=np.float64)
    means = img.mean(axis=1, keepdims=True)
    normalized = np.where(means > 0, img / np.where(means > 0, means, 1.0), img)
    return image_adjust(normalized.astype(_F32).astype(np.float64), None, real32=True)


def normalize_lines(img: np.ndarray) -> np.ndarray:
    """Normalize each row of the kymograph by its mean intensity, then stretch to [0, 1].

    Rows with mean 0 are left unchanged. Corresponds to normlines in KymoButler.wl line 44.
    Returns float32.
    """
    return _normlines(img).astype(np.float32)


def color_negate(raw: np.ndarray, *, real32: bool = False) -> np.ndarray:
    """``ColorNegate``: ``1 - x`` (on the pixel grid for integer types, in float32 for Real32)."""
    if real32:
        return (_F32(1) - np.asarray(raw, dtype=_F32)).astype(np.float64)
    return 1.0 - raw


def preprocess_rawkym(raw: np.ndarray, *, real32: bool = False) -> tuple[np.ndarray, bool]:
    """``negkym = normlines@If[isNegated@rawkym, ColorNegate@rawkym, rawkym]`` (float64 array
    of float32 values). ``real32``: rawkym is a Real32 image."""
    negated = is_negated(raw)
    return _normlines(color_negate(raw, real32=real32) if negated else raw), negated


def net_input_shape(shape: tuple[int, int]) -> tuple[int, int]:
    """``16*Round@N[dim/16]`` (Round is half to even).

    Raises:
        ValueError: if a side is 8 px or less, where Mathematica's size is 0 and the network
            cannot run.
    """
    out = tuple(16 * int(round(d / 16)) for d in shape[:2])
    if min(out) < 16:
        raise ValueError(
            f"kymograph of {shape[0]}x{shape[1]} px is too small: KymoButler needs at least "
            "9 px on each side (16*Round[d/16] would be 0)"
        )
    return out  # type: ignore[return-value]


# ------------------------------------------------------------------------------ ImageResize
# ImageResize with the default Resampling -> Automatic uses a box kernel ("Nearest" when
# enlarging) when neither side of the input image exceeds this many pixels, and "Lanczos"
# otherwise, whatever the output size and the pixel type (Mathematica 14.3: no exception in
# 19,543 Real32 resizes of every w x h <= 40 x 40 to 12 output sizes and of larger images, in
# 192 resizes of Byte/Bit16/Real32/Real64 images around the 24/25 px boundary and in a fresh set
# of 176 resizes of the four types, 44 shapes from 1 x 1 to 301 x 37 px, e.g. 24 x 24 -> 25 x 25
# is nearest neighbour and 25 x 1, 1 x 25, 64 x 24 -> 64 x 64 are Lanczos; a third fresh set of
# 152 resizes, 1 x 1 to 300 x 3 and 24/25 px sides of all four types to 64 x 64, agrees too).
_SMALL_IMAGE_MAX_SIDE = 24
# Pixels whose plain double-precision value is this close to a rounding boundary of the pixel
# type are recomputed in Mathematica's exact order of operations (``_resample_axis``). The
# bound on the difference of the two computations is ~(taps) * 2^-52 * sum|w_t x_t|: under
# 1e-8 grid levels for Byte/Bit16 images; for Real32 it is computed per pixel.
_LEVEL_TIE_TOLERANCE = 1e-6


def _lanczos_kernel(x: np.ndarray, a: int = 3) -> np.ndarray:
    out = np.sinc(x) * np.sinc(x / a)
    out[np.abs(x) >= a] = 0.0
    return out


def _reflect(ks: np.ndarray, n: int) -> np.ndarray:
    """Symmetric (half-sample) reflection, repeated as often as needed: -1 -> 0, n -> n - 1."""
    ks = np.mod(ks, 2 * n)
    return np.where(ks >= n, 2 * n - 1 - ks, ks)


def _check_sizes(n_in: int, n_out: int) -> None:
    if n_in < 1 or n_out < 1:
        raise ValueError(f"cannot resample {n_in} px to {n_out} px")


def lanczos_taps(n_in: int, n_out: int, a: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Mathematica's ``Resampling -> "Lanczos"`` (Lanczos 3) along one axis, as taps.

    Returns ``(idx, w)``, two ``(n_out, width)`` arrays: output pixel ``j`` is
    ``sum_t w[j, t] * x[idx[j, t]] / sum_t w[j, t]``, the taps in this order. Output pixel ``j``
    is centred at ``c = (j + 1/2) * s - 1/2`` input pixels (``s = n_in / n_out``); the kernel is
    widened by ``max(s, 1)``; indices outside the image reflect symmetrically (-1 -> 0,
    n -> n - 1), repeatedly for images narrower than the kernel (exact for 1, 2 and 3 px sides).
    """
    _check_sizes(n_in, n_out)
    s = n_in / n_out
    fs = max(s, 1.0)
    c = (np.arange(n_out) + 0.5) * s - 0.5
    lo = np.floor(c - a * fs).astype(np.int64)
    width = int(np.ceil(2 * a * fs)) + 2
    ks = lo[:, None] + np.arange(width)[None, :]
    return _reflect(ks, n_in), _lanczos_kernel((ks - c[:, None]) / fs, a)


def box_taps(n_in: int, n_out: int) -> tuple[np.ndarray, np.ndarray]:
    """The box kernel ``Resampling -> Automatic`` uses for small images, as taps (see
    ``lanczos_taps``), on the same grid: input pixel ``k`` has weight 1 if ``|d| < 1/2`` and
    1/2 if ``|d| = 1/2``, ``d = (k - c) / max(s, 1)``, computed in double precision exactly like
    that. Enlarging this is nearest neighbour (the two neighbours averaged on an exact tie),
    shrinking the mean of the pixels in the box.

    Exact whenever ``s`` is a binary fraction (every resize KymoButler makes: to 16, 32 or 64
    px). Of 430 probed box-kernel resizes of all four pixel types (Mathematica 14.3, sides 1 to
    24 px, to 1 to 127 px) 408 are bit for bit; the other 22 have a scale that is not a binary
    fraction and a box edge or a nearest-neighbour midpoint exactly on a pixel centre, where
    Mathematica includes, excludes or averages that pixel depending on rounding details not
    reproduced here (one row or column each; e.g. the middle pixel of 13 -> 10 and 17 -> 10, or
    10 -> 77 at output pixel 38). 17 alternative formulas for the centres and edges, in double
    and single precision, fit fewer of them.
    """
    _check_sizes(n_in, n_out)
    s = n_in / n_out
    fs = max(s, 1.0)
    c = (np.arange(n_out) + 0.5) * s - 0.5
    d = np.abs((np.arange(n_in)[None, :] - c[:, None]) / fs)
    w = np.where(d < 0.5, 1.0, np.where(d == 0.5, 0.5, 0.0))
    width = int((w > 0).sum(axis=1).max())
    idx = np.argsort(w == 0, axis=1, kind="stable")[:, :width]  # the pixels in the box, in order
    return idx, np.take_along_axis(w, idx, axis=1)


def _taps_matrix(idx: np.ndarray, w: np.ndarray, n_in: int, normalise: bool) -> sp.csr_matrix:
    n_out, width = w.shape
    m = sp.csr_matrix(
        (w.ravel(), (np.repeat(np.arange(n_out), width), idx.ravel())), shape=(n_out, n_in)
    )
    m.sum_duplicates()
    if normalise:
        m = sp.csr_matrix(sp.diags(1.0 / np.asarray(m.sum(axis=1)).ravel()) @ m)
    return m


def lanczos_matrix(n_in: int, n_out: int, a: int = 3) -> sp.csr_matrix:
    """``lanczos_taps`` as a sparse ``(n_out, n_in)`` matrix with rows normalised to sum 1."""
    return _taps_matrix(*lanczos_taps(n_in, n_out, a), n_in, normalise=True)


def nearest_matrix(n_in: int, n_out: int) -> sp.csr_matrix:
    """``box_taps`` as a sparse ``(n_out, n_in)`` matrix with rows normalised to sum 1."""
    return _taps_matrix(*box_taps(n_in, n_out), n_in, normalise=True)


def _resample_axis0(
    x: np.ndarray, taps: tuple[np.ndarray, np.ndarray], levels: int | None, real32: bool
) -> np.ndarray:
    """One ImageResize pass along axis 0 of the C-contiguous 2-D array ``x`` (integer levels
    for Byte/Bit16).

    Mathematica sums ``w_t * x_t`` over the taps in order with fused multiply-adds in double
    precision and divides by the sum of the weights, ``(((w_1 x_1) + w_2 x_2) + ...) / ((w_1 +
    w_2) + ...)``; a Byte/Bit16 result is then clipped and rounded half up onto the levels, a
    Real32 one rounded to float32. The order of operations matters only where the result is,
    within rounding, a tie of the type's rounding (n + 1/2 levels, or a float32 midpoint):
    those pixels are recomputed exactly that way, all others use a sparse matrix product. The
    two computations differ by at most ``(2 width + 1) 2^-53 sum_t |w_t x_t| / sum_t w_t``;
    the tolerances below are twice that bound.
    """
    idx, w = taps
    n_out, width = w.shape
    den = np.zeros(n_out)
    for t in range(width):
        den = den + w[:, t]
    m = _taps_matrix(idx, w, x.shape[0], normalise=False)
    y = m @ x
    y /= den[:, None]
    gain = float((np.abs(w).sum(axis=1) / np.abs(den)).max())  # bounds sum|w_t x_t| / den / max|x|
    tol = (width + 2) * 2.0**-51 * gain
    if levels is not None:
        cand = np.abs(y - np.floor(y) - 0.5) <= tol * levels
    elif real32:
        g = tol * float(np.abs(x).max())
        cand = (y - g).astype(_F32) != (y + g).astype(_F32)
    else:
        cand = None
    if cand is not None and cand.any():
        j, c = np.nonzero(cand)
        xs = x[idx[j], c[:, None]]
        ws = w[j]
        if real32:  # the per-pixel bound
            g = (width + 2) * 2.0**-51 * np.abs(ws * xs).sum(axis=1) / np.abs(den[j])
            keep = (y[j, c] - g).astype(_F32) != (y[j, c] + g).astype(_F32)
            j, c, xs, ws = j[keep], c[keep], xs[keep], ws[keep]
        acc = np.zeros(len(j))
        for t in range(width):
            acc = _fma64(ws[:, t], xs[:, t], acc)
        y[j, c] = acc / den[j]
    if levels is not None:
        np.clip(y, 0.0, levels, out=y)
        return np.floor(y + 0.5)
    if real32:
        return y.astype(_F32).astype(np.float64)
    return y


def image_resize(
    x: np.ndarray, out_shape: tuple[int, int], levels: int | None = None, *, real32: bool = False
) -> np.ndarray:
    """``ImageResize[img, {w, h}]`` with the default resampling, as float64.

    Resampling -> Automatic is "Lanczos" (``lanczos_taps``) unless neither input side exceeds
    24 px, where it is a box kernel (``box_taps``). The image is resized along x first and then
    along y (a side that keeps its size is not resampled); each pass is computed as
    ``_resample_axis0`` describes and has the image's pixel type. Mathematica 14.3, bit for bit
    on every Byte, Bit16 and Real32 probe: 339 Lanczos resizes (2.1 M pixels, including step-edge
    data rich in exact ties, e.g. 128 -> 64 and 40 -> 100, where 1,384 pixels come out 1 level
    or 1 float32 ulp off when the pass is computed with normalised weights instead) and 157
    box-kernel resizes except for the box-edge ties ``box_taps`` describes.

    Args:
        x: 2-D image (float, [0, 1] scale for integer types).
        out_shape: (height, width).
        levels: None for a Real image (no clipping). 255 / 65535 for a Byte / Bit16 image (on
            its grid): each pass is computed on the integer levels, clipped and rounded half up.
        real32: for a Real image, whether it is Real32: each pass is rounded to float32. A
            Real64 image is computed in double precision: within a few ulp of Mathematica's
            (its accumulation order for Real64 is not reproduced).

    Raises:
        ValueError: for a Bit image (levels 1): Mathematica resizes binary images differently
            (``kymobutler.morphology.resize_binary``).
    """
    if levels == 1:
        raise ValueError("image_resize does not resize Bit images; see morphology.resize_binary")
    x = np.asarray(x, dtype=np.float64)
    h, w = int(out_shape[0]), int(out_shape[1])
    _check_sizes(min(x.shape), min(h, w))
    taps = box_taps if max(x.shape) <= _SMALL_IMAGE_MAX_SIDE else lanczos_taps
    t = np.rint(x * levels) if levels is not None else x
    if t.shape[1] != w:
        t = _resample_axis0(np.ascontiguousarray(t.T), taps(t.shape[1], w), levels, real32).T
    if t.shape[0] != h:
        t = _resample_axis0(np.ascontiguousarray(t), taps(t.shape[0], h), levels, real32)
    if levels is not None:
        t = t / levels
    return np.ascontiguousarray(t) if t is not x else x.copy()


def resize_to_multiple_of_16(img: np.ndarray) -> np.ndarray:
    """``ImageResize[negkym, 16*Round@N[dim/16]]`` (KymoButler.wl line 55), the network input.

    Default resampling of the Real32 image ``negkym`` (see ``image_resize``); identity when
    both sides are already multiples of 16. Returns float32.
    """
    return image_resize(img, net_input_shape(img.shape), real32=True).astype(np.float32)


def load_and_preprocess_mma(
    image_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, bool, int | None]:
    """float64 ``(negkym, rawkym, was_negated, levels)`` of a kymograph file (see module doc);
    ``levels`` is rawkym's pixel type (``rawkym_levels``)."""
    raw, levels, real32 = _load_rawkym(image_path)
    preprocessed, negated = preprocess_rawkym(raw, real32=real32)
    return preprocessed, raw, negated, levels


def load_and_preprocess(image_path: str | Path) -> tuple[np.ndarray, np.ndarray, bool]:
    """Full preprocessing pipeline for a kymograph image.

    Steps (Mathematica's, see the module docstring): import, drop alpha, per-channel
    ImageAdjust, grayscale, ImageAdjust, polarity detection (white background -> invert),
    per-row normalisation.

    Args:
        image_path: Path to the kymograph image.

    Returns:
        Tuple of (preprocessed_image, raw_grayscale, was_negated).
        - preprocessed_image: ``negkym``, float32 HxW in [0, 1]
        - raw_grayscale: ``rawkym``, float32 HxW in [0, 1] (before negation/normalization)
        - was_negated: True if background was white and image was inverted

    Raises:
        UnsupportedImageFormat: see ``import_image``.
    """
    preprocessed, raw, negated, _ = load_and_preprocess_mma(image_path)
    return preprocessed.astype(np.float32), raw.astype(np.float32), negated
