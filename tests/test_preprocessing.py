"""Tests for preprocessing module."""

import json
from pathlib import Path

import numpy as np
import pytest

from kymobutler.preprocessing import (
    is_negated,
    normalize_lines,
    resize_to_multiple_of_16,
    load_and_preprocess,
)


class TestIsNegated:
    def test_white_background(self):
        # Mostly white image
        img = np.ones((50, 50), dtype=np.float32) * 0.9
        img[20:30, 20:30] = 0.1
        assert is_negated(img) is True

    def test_black_background(self):
        # Mostly black image
        img = np.ones((50, 50), dtype=np.float32) * 0.1
        img[20:30, 20:30] = 0.9
        assert is_negated(img) is False


class TestNormalizeLines:
    def test_uniform_rows_unchanged(self):
        img = np.ones((10, 10), dtype=np.float32) * 0.5
        result = normalize_lines(img)
        assert result.shape == (10, 10)
        assert result.dtype == np.float32

    def test_zero_rows_preserved(self):
        img = np.zeros((10, 10), dtype=np.float32)
        img[5] = 0.5
        result = normalize_lines(img)
        assert np.all(result[0] == 0.0)  # zero row stays zero


class TestResizeToMultipleOf16:
    def test_already_multiple(self):
        img = np.zeros((64, 128), dtype=np.float32)
        result = resize_to_multiple_of_16(img)
        assert result.shape == (64, 128)

    def test_needs_resize(self):
        img = np.zeros((60, 100), dtype=np.float32)
        result = resize_to_multiple_of_16(img)
        assert result.shape[0] % 16 == 0
        assert result.shape[1] % 16 == 0

    def test_small_image(self):
        img = np.zeros((10, 10), dtype=np.float32)
        result = resize_to_multiple_of_16(img)
        assert result.shape[0] >= 16
        assert result.shape[1] >= 16


class TestLoadAndPreprocess:
    def test_loads_png(self, unitest_path):
        preprocessed, raw, was_negated = load_and_preprocess(unitest_path)
        assert preprocessed.ndim == 2
        assert raw.ndim == 2
        assert preprocessed.dtype == np.float32
        assert 0 <= preprocessed.min()
        assert preprocessed.max() <= 1.0
        assert isinstance(was_negated, bool)

    def test_output_shapes_match(self, bitest_path):
        preprocessed, raw, _ = load_and_preprocess(bitest_path)
        assert preprocessed.shape == raw.shape

    def test_float_tiff_not_collapsed_to_black(self, tmp_path):
        # Regression: 16-bit-range float TIFFs (e.g. microscope exports) used to hit
        # convert("L")'s >255 clamp, collapsing to a constant image that polarity detection
        # inverted to solid black -> zero tracks. The gradient must survive with real range.
        from PIL import Image

        arr = np.linspace(800.0, 26000.0, 64 * 96, dtype=np.float32).reshape(64, 96)
        p = tmp_path / "scan_16bit.tif"
        Image.fromarray(arr).save(str(p))
        preprocessed, raw, _ = load_and_preprocess(str(p))
        assert float(preprocessed.max()) > 0.5          # signal preserved, not black
        assert float(raw.min()) < float(raw.max())      # dynamic range retained

    def test_8bit_png_full_range(self, tmp_path):
        # The mode branch must leave ordinary 8-bit images behaving as before.
        from PIL import Image

        arr = (np.arange(64 * 96, dtype=np.int64) % 256).astype(np.uint8).reshape(64, 96)
        p = tmp_path / "img.png"
        Image.fromarray(arr).save(str(p))
        preprocessed, raw, _ = load_and_preprocess(str(p))
        assert preprocessed.shape == (64, 96)
        assert 0.0 <= float(preprocessed.min()) and float(preprocessed.max()) <= 1.0


class TestMathematicaImageArithmetic:
    """Values from Mathematica 14.3 probes (ImageAdjust / ColorConvert on synthetic images)."""

    @pytest.mark.parametrize(
        "lo,k,hi,want",
        [(24, 130, 236, 127), (0, 1, 2, 128), (100, 101, 102, 127), (8, 80, 248, 77),
         (8, 128, 248, 128), (50, 56, 70, 76)],
    )
    def test_image_adjust_byte_ties(self, lo, k, hi, want):
        from kymobutler.preprocessing import image_adjust

        out = image_adjust(np.array([[lo, k, hi]]) / 255.0, 255)
        assert np.rint(out[0, 1] * 255) == want

    @pytest.mark.parametrize(
        "lo,k,hi,want",
        [(1000, 1200, 3000, 6554), (381, 16614, 32847, 32767), (0, 32767, 65534, 32768),
         (49677, 49830, 65283, 642), (8432, 8434, 8500, 1927), (8432, 8450, 8500, 17348)],
    )
    def test_image_adjust_bit16_ties(self, lo, k, hi, want):
        from kymobutler.preprocessing import image_adjust

        out = image_adjust(np.array([[lo, k, hi]]) / 65535.0, 65535)
        assert np.rint(out[0, 1] * 65535) == want

    def test_image_adjust_constant_is_zero(self):
        from kymobutler.preprocessing import image_adjust

        assert not image_adjust(np.full((2, 3), 0.3), None).any()
        assert not image_adjust(np.full((2, 3), 77 / 255), 255).any()

    def test_grayscale_matches_mathematica(self):
        from kymobutler.preprocessing import color_convert_grayscale

        rgb = np.array(
            [[[50, 140, 205], [246, 160, 9], [144, 170, 111], [162, 188, 129], [221, 35, 34]]]
        )
        gray = color_convert_grayscale(rgb / 255.0, 255)
        assert np.rint(gray[0] * 255).tolist() == [121, 169, 156, 174, 90]

    def test_alpha_is_dropped_not_composited(self, tmp_path):
        from PIL import Image

        from kymobutler.preprocessing import load_grayscale

        rng = np.random.default_rng(0)
        rgb = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
        alpha = rng.integers(0, 256, (20, 30, 1), dtype=np.uint8)
        Image.fromarray(np.concatenate([rgb, alpha], axis=2)).save(tmp_path / "a.png")
        Image.fromarray(rgb).save(tmp_path / "b.png")
        raw_a, lv = load_grayscale(tmp_path / "a.png")
        raw_b, _ = load_grayscale(tmp_path / "b.png")
        assert lv == 255 and np.array_equal(raw_a, raw_b)

    def test_rawkym_on_byte_grid(self, unitest_path):
        from kymobutler.preprocessing import load_grayscale

        raw, lv = load_grayscale(unitest_path)
        assert lv == 255 and np.allclose(raw * 255, np.rint(raw * 255), atol=1e-9)

    def test_16bit_colour_tiff_read_at_full_depth(self, tmp_path):
        tifffile = pytest.importorskip("tifffile")
        from kymobutler.preprocessing import read_image

        rgb = np.random.default_rng(1).integers(0, 65536, (12, 17, 3)).astype(np.uint16)
        tifffile.imwrite(tmp_path / "c.tif", rgb, photometric="rgb")
        pixels, lv = read_image(tmp_path / "c.tif")
        assert lv == 65535 and np.array_equal(np.rint(pixels * 65535), rgb)

    def test_16bit_colour_png_read_at_full_depth(self, tmp_path):
        # Pillow alone keeps only the high bytes; Mathematica imports Bit16
        import struct
        import zlib

        from kymobutler.preprocessing import read_image

        a = np.random.default_rng(2).integers(0, 65536, (5, 7, 3)).astype(">u2")
        raw = b"".join(b"\x00" + a[i].tobytes() for i in range(a.shape[0]))

        def chunk(tag, data):
            crc = struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
            return struct.pack(">I", len(data)) + tag + data + crc

        ihdr = struct.pack(">IIBBBBB", 7, 5, 16, 2, 0, 0, 0)
        (tmp_path / "c.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b"")
        )
        pixels, lv = read_image(tmp_path / "c.png")
        assert lv == 65535 and np.array_equal(np.rint(pixels * 65535), a.astype(np.int64))

    def test_cmyk_tiff_is_read_as_rgb(self, tmp_path):
        from PIL import Image

        from kymobutler.preprocessing import read_image

        cmyk = np.random.default_rng(3).integers(0, 256, (6, 8, 4)).astype(np.uint8)
        Image.frombytes("CMYK", (8, 6), cmyk.tobytes()).save(tmp_path / "c.tif")
        pixels, lv = read_image(tmp_path / "c.tif")
        c, k = cmyk[..., :3] / 255.0, cmyk[..., 3:] / 255.0
        assert lv == 255 and np.array_equal(pixels, np.round((1 - c) * (1 - k) * 255) / 255)

    def test_unsupported_formats_raise_a_clear_error(self, tmp_path):
        from PIL import Image

        from kymobutler.preprocessing import UnsupportedImageFormat, read_image

        frames = [Image.fromarray(np.full((4, 5), v, np.uint8)) for v in (10, 20)]
        frames[0].save(tmp_path / "stack.tif", save_all=True, append_images=frames[1:])
        with pytest.raises(UnsupportedImageFormat, match=r"^stack\.tif: the file holds 2 images"):
            read_image(tmp_path / "stack.tif")
        Image.fromarray(np.ones((4, 5), bool)).save(tmp_path / "bits.tif")  # no BitsPerSample
        with pytest.raises(NotImplementedError, match="BitsPerSample"):
            read_image(tmp_path / "bits.tif")
        Image.new("YCbCr", (4, 5)).save(tmp_path / "ycc.jpg")  # JPEG YCbCr is decoded to RGB
        assert read_image(tmp_path / "ycc.jpg")[0].shape == (5, 4, 3)


REF = np.load(Path(__file__).parent / "data" / "reference" / "preprocessing_mma.npz")
REF_META = json.loads(
    (Path(__file__).parent / "data" / "reference" / "preprocessing_mma.json").read_text()
)
LEVELS = {"Byte": 255, "Bit16": 65535, "Real32": None, "Real64": None}


def _values(key, image_type):
    lv = LEVELS[image_type]
    a = REF[key].astype(np.float64)
    return a / lv if lv else a


class TestMathematicaProbes:
    """Exact agreement with Mathematica 14.3 on synthetic probes (preprocessing_mma.json)."""

    def test_grayscale_byte(self):
        from kymobutler.preprocessing import color_convert_grayscale

        rgb = REF["gray_byte_rgb"].astype(np.float64)[None] / 255
        out = np.rint(color_convert_grayscale(rgb, 255)[0] * 255)
        assert np.array_equal(out, REF["gray_byte_mma"])  # 2,500 of the 4,000 are exact ties

    def test_grayscale_bit16(self):
        from kymobutler.preprocessing import color_convert_grayscale

        rgb = REF["gray_b16_rgb"].astype(np.float64)[None] / 65535
        out = np.rint(color_convert_grayscale(rgb, 65535)[0] * 65535)
        assert np.array_equal(out, REF["gray_b16_mma"])

    def test_grayscale_real32_and_real64(self):
        from kymobutler.preprocessing import color_convert_grayscale

        out32 = color_convert_grayscale(REF["gray_r32_rgb"].astype(np.float64)[None], None,
                                        real32=True)[0]
        assert np.array_equal(out32, REF["gray_r32_mma"].astype(np.float64))
        out64 = color_convert_grayscale(REF["gray_r64_rgb"][None], None)[0]
        assert np.array_equal(out64, REF["gray_r64_mma"])

    @pytest.mark.parametrize("case", REF_META["adjust"], ids=lambda c: f"{c['type']}_{c['j']}")
    def test_image_adjust(self, case):
        from kymobutler.preprocessing import image_adjust

        t = case["type"]
        x = _values(f"adjust_{case['j']}_in", t)
        out = image_adjust(x, LEVELS[t], real32=t == "Real32")
        assert np.array_equal(out, _values(f"adjust_{case['j']}_mma", t))

    def test_image_adjust_bit16_every_level(self):
        from kymobutler.preprocessing import image_adjust

        got = [np.rint(image_adjust(np.arange(a, b + 1)[None] / 65535, 65535)[0] * 65535)
               for a, b in REF["adjust_b16_ranges"]]
        assert np.array_equal(np.concatenate(got), REF["adjust_b16_mma"])

    @pytest.mark.parametrize("case", REF_META["resize"], ids=lambda c: c["key"])
    def test_image_resize(self, case):
        from kymobutler.preprocessing import image_resize

        t = case["type"]
        x = _values(case["key"] + "_in", t)
        out = image_resize(x, tuple(case["out"]), LEVELS[t], real32=t == "Real32")
        assert np.array_equal(out, _values(case["key"] + "_mma", t))


class TestImageAdjustParallelMinMax:
    """Mathematica's ImageAdjust ignores the last pixel in the min/max of an image of n > 8192
    pixels with n = 1 (mod 4) (probed: every n in 4097..12289, and every position of the
    extreme): here the planted maximum 5000 at the last pixel is ignored for n = 9025."""

    @staticmethod
    def _adjust(n, levels):
        from kymobutler.preprocessing import image_adjust

        v = (1000 + np.arange(1, n) % 1000).astype(np.float64)
        if levels == 255:
            v = 10 + np.arange(1, n) % 100.0
        x = np.append(v, 5000 if levels != 255 else 250)[None] / (levels or 1)
        return image_adjust(x, levels)[0]

    @pytest.mark.parametrize("levels", [255, 65535, None])
    def test_last_pixel_ignored(self, levels):
        out = self._adjust(9025, levels)
        assert out[:-1].max() == 1.0 and out[-1] == 1.0  # the rest is stretched to [0, 1]

    @pytest.mark.parametrize("n", [9024, 9026, 9027, 8189])
    def test_other_sizes_unaffected(self, n):
        out = self._adjust(n, 65535)
        assert out[-1] == 1.0 and out[:-1].max() < 0.5


class TestImageResizeRules:
    def test_small_images_use_the_box_kernel(self):
        from kymobutler.preprocessing import image_resize

        x = np.random.default_rng(1).random((24, 24))
        out = image_resize(x, (64, 64))
        idx = np.floor((np.arange(64) + 0.5) * 24 / 64).astype(int)  # nearest neighbour
        assert np.array_equal(out, x[np.ix_(idx, idx)])
        y = np.random.default_rng(2).random((24, 25))  # a side over 24 px: Lanczos
        assert not np.isin(image_resize(y, (64, 64)), y).all()

    def test_tiny_sides_resample(self):
        from kymobutler.preprocessing import image_resize

        for shape in ((1, 100), (2, 100), (3, 100), (100, 1), (1, 1)):
            x = np.random.default_rng(0).integers(0, 256, shape) / 255
            out = image_resize(x, (64, 64), 255)
            assert out.shape == (64, 64) and np.array_equal(np.rint(out * 255), out * 255)

    def test_real32_passes_are_float32(self):
        from kymobutler.preprocessing import image_resize

        x = np.random.default_rng(3).random((40, 90)).astype(np.float32).astype(np.float64)
        out = image_resize(x, (48, 96), real32=True)
        assert np.array_equal(out, out.astype(np.float32).astype(np.float64))


class TestLanczosResize:
    def test_rows_sum_to_one_and_identity(self):
        from kymobutler.preprocessing import image_resize, lanczos_matrix

        for n_in, n_out in ((83, 80), (92, 96), (670, 64), (9, 16)):
            m = lanczos_matrix(n_in, n_out)
            assert np.allclose(np.asarray(m.sum(axis=1)).ravel(), 1.0)
        x = np.random.default_rng(0).random((32, 48))
        assert np.array_equal(image_resize(x, (32, 48)), x)

    def test_net_input_shape(self):
        from kymobutler.preprocessing import net_input_shape

        assert net_input_shape((83, 92)) == (80, 96)
        assert net_input_shape((24, 40)) == (32, 32)  # Round is half to even: 1.5 -> 2, 2.5 -> 2
        assert net_input_shape((9, 100)) == (16, 96)
        with pytest.raises(ValueError):
            net_input_shape((8, 100))  # 16*Round[8/16] = 0
