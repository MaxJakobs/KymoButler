"""Tests for morphological processing module."""

import hashlib
import itertools
from pathlib import Path

import numpy as np
import pytest

from kymobutler.morphology import (
    smooth_binary_uni,
    smooth_binary_bi,
    chew_ends,
    chew_all_ends,
    detect_seeds,
    _filter_components,
    _select_components_uni,
    morphological_components,
    prune_skeleton,
    resize_binary,
    process_segmentation_uni,
    process_segmentation_bi,
    check_image_size,
)

DATA = Path(__file__).parent / "data"


class TestSmoothBinaryUni:
    def test_fills_gaps(self):
        # Create an L-shaped gap that should be filled
        binary = np.zeros((5, 5), dtype=bool)
        binary[1, 2] = True
        binary[2, 3] = True
        binary[3, 2] = True
        result = smooth_binary_uni(binary)
        # Should have filled the center
        assert result[2, 2]

    def test_no_change_on_empty(self):
        binary = np.zeros((5, 5), dtype=bool)
        result = smooth_binary_uni(binary)
        assert not np.any(result)


class TestSmoothBinaryBi:
    def test_output_is_binary(self):
        rng = np.random.default_rng(42)
        binary = rng.random((20, 20)) > 0.7
        result = smooth_binary_bi(binary)
        assert result.dtype == bool


class TestChewEnds:
    def test_removes_horizontal_endpoints(self):
        binary = np.zeros((5, 7), dtype=bool)
        binary[2, 1:6] = True  # horizontal line
        result = chew_ends(binary)
        # Endpoints at (2,1) and (2,5) should be removed
        assert not result[2, 1]
        assert not result[2, 5]
        # Interior points preserved
        assert result[2, 3]

    def test_chew_all_ends_converges(self):
        binary = np.zeros((5, 7), dtype=bool)
        binary[2, 1:6] = True
        result = chew_all_ends(binary)
        # After chewing all ends, nothing should remain for a pure horizontal line
        # (each iteration removes endpoints until nothing left)
        assert not np.any(result) or np.sum(result) <= 1


class TestDetectSeeds:
    def test_finds_top_endpoints(self, simple_skeleton):
        seeds = detect_seeds(simple_skeleton)
        assert len(seeds) > 0
        # Seeds should be sorted by row
        for i in range(len(seeds) - 1):
            assert seeds[i][0] <= seeds[i + 1][0]


class TestPruneSkeletonShortBranches:
    def test_removes_short_branches_keeps_line(self, simple_skeleton):
        pruned = prune_skeleton(simple_skeleton, 5)
        # The short branch at row 15 should be removed
        assert np.sum(pruned) < np.sum(simple_skeleton)
        # Main vertical line should still exist
        assert np.any(pruned[10:20, 16])


class TestFilterComponents:
    def test_removes_small_components(self):
        binary = np.zeros((50, 50), dtype=bool)
        # Large component
        binary[5:30, 25] = True
        # Small component
        binary[40, 10:12] = True
        result = _filter_components(binary, min_size=5, min_frames=5)
        # Large component should remain
        assert np.any(result[5:30, 25])
        # Small component should be removed
        assert not np.any(result[40, 10:12])

    def test_frame_span_is_bounding_box_height(self):
        # Mathematica: #BoundingBox[[2,2]] - #BoundingBox[[1,2]] is the number of rows spanned
        binary = np.zeros((30, 30), dtype=bool)
        binary[10:15, 5] = True  # 5 rows
        assert _filter_components(binary, min_size=5, min_frames=5).sum() == 5
        assert _filter_components(binary, min_size=5, min_frames=6).sum() == 0
        assert _filter_components(binary, min_size=6, min_frames=5).sum() == 0



    def test_single_pixels_survive_only_with_min_size_and_min_frames_at_most_one(self):
        # the only way to get a seed without skeleton neighbours in BiKymoButlerTrack
        binary = np.zeros((20, 20), dtype=bool)
        binary[3, 3] = True
        binary[8:18, 10] = True
        assert _filter_components(binary, 1, 1)[3, 3]
        assert _filter_components(binary, 0, 0)[3, 3]
        assert not _filter_components(binary, 2, 1)[3, 3]
        assert not _filter_components(binary, 1, 2)[3, 3]
        assert _filter_components(binary, 2, 1)[8:18, 10].all()


class TestSelectComponentsUni:
    """SelectComponents[., #Length >= minSz &&
    #BoundingBox[[2,2]] - #BoundingBox[[1,2]] >= minFr &]."""

    def test_length_of_lines(self):
        # Length = 4 sqrt(largest covariance eigenvalue): a vertical line of n px has (n^2-1)/12
        binary = np.zeros((30, 30), dtype=bool)
        binary[10:17, 5] = True  # 7 px -> Length exactly 8
        assert _select_components_uni(binary, 8, 7).sum() == 7
        assert _select_components_uni(binary, 8.01, 7).sum() == 0
        assert _select_components_uni(binary, 3, 8).sum() == 0  # spans 7 rows
        pair = np.zeros((5, 5), dtype=bool)
        pair[2, 1:3] = True  # 2 px -> Length exactly 2
        assert _select_components_uni(pair, 2, 1).sum() == 2
        assert _select_components_uni(pair, 2.01, 1).sum() == 0
        dot = np.zeros((5, 5), dtype=bool)
        dot[2, 2] = True  # a single pixel has Length 0.5
        assert _select_components_uni(dot, 0.5, 1).sum() == 1
        assert _select_components_uni(dot, 1, 1).sum() == 0

    def test_length_not_pixel_count(self):
        # a compact 3x3 blob has 9 px but Length 4*sqrt(2/3) = 3.27
        blob = np.zeros((10, 10), dtype=bool)
        blob[3:6, 3:6] = True
        assert _select_components_uni(blob, 3, 3).sum() == 9
        assert _select_components_uni(blob, 4, 3).sum() == 0


class TestMorphologicalComponents:
    def test_labels_and_count(self):
        b = np.zeros((6, 6), dtype=bool)
        b[0, 0] = b[1, 1] = True  # 8-connected
        b[4, 4:6] = True
        labels, n = morphological_components(b)
        assert n == 2
        assert labels[0, 0] == labels[1, 1] == 1 and labels[4, 4] == labels[4, 5] == 2
        assert (labels[~b] == 0).all()

    # Expected label images below are Mathematica 14.3's MorphologicalComponents output.

    def test_tie_with_west_neighbour_goes_to_the_north_east_set(self):
        # (2,4) joins the left arm (its W, provisional 1) and the right arm (its NE, 3): equal
        # sizes, so the right arm's root wins and the dot (2) is numbered first
        b = np.array([[1, 0, 0, 1, 0, 0, 1],
                      [0, 1, 0, 0, 0, 1, 0],
                      [0, 0, 1, 1, 1, 0, 0]], dtype=bool)
        want = [[2, 0, 0, 1, 0, 0, 2],
                [0, 2, 0, 0, 0, 2, 0],
                [0, 0, 2, 2, 2, 0, 0]]
        labels, n = morphological_components(b)
        assert n == 2 and labels.tolist() == want

    def test_tie_with_north_west_neighbour_goes_to_the_north_west_set(self):
        # a V: (3,3) has W background, so its first labelled neighbour is NE (right arm, 3) and
        # NW (left arm, 1) wins the tie: first-pixel order
        b = np.zeros((5, 8), dtype=bool)
        for i in range(4):
            b[i, i] = b[i, 6 - i] = True
        b[0, 3] = True
        labels, n = morphological_components(b)
        assert n == 2 and labels[0, 0] == 1 and labels[0, 3] == 2

    def test_union_by_number_of_labels(self):
        # the left arm already holds two provisional labels (1 and 2) when it meets the right
        # arm (4) at (3,5): the larger set's root (1) wins although the tie rule would pick 4
        b = np.array([[1, 0, 1, 0, 0, 1, 0, 0, 1],
                      [0, 1, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 1, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 1, 1, 1, 0, 0, 0]], dtype=bool)
        labels, n = morphological_components(b)
        assert n == 2 and labels[0, 0] == labels[0, 8] == 1 and labels[0, 5] == 2

    def test_matches_mathematica_on_synthetic_images(self):
        # 110 small synthetic images (random, thinned, blobs, dilated skeletons, combs, walks)
        # where Mathematica's numbering differs from first-pixel order or from the earlier port
        data = np.load(DATA / "mma_morphological_components.npz")
        bits = np.unpackbits(data["bits"]).astype(bool)
        i = 0
        for h, w in data["shapes"]:
            b = bits[i:i + h * w].reshape(h, w)
            want = data["labels"][i:i + h * w].reshape(h, w)
            i += h * w
            labels, n = morphological_components(b)
            assert np.array_equal(labels, want) and n == want.max()
        assert len(data["shapes"]) == 110

    def test_matches_mathematica_on_fresh_images(self):
        # 146 images of a set generated independently of the implementation (random images,
        # strips, thinned and skeletonized blobs and noise, diagonal segments, V cascades,
        # checkerboards, the uni and bi chains, skeletons and straddler dilations of the public
        # test images) whose Mathematica 14.3 numbering differs from first-pixel order. Stored
        # per image: Mathematica's label of each component, components in first-pixel order.
        from scipy.ndimage import label

        data = np.load(DATA / "mma_morphological_components_order.npz")
        bits = np.unpackbits(data["bits"]).astype(bool)
        i = j = 0
        for (h, w), n in zip(data["shapes"], data["ncomp"]):
            b = bits[i:i + h * w].reshape(h, w)
            want = data["perm"][j:j + n]
            i, j = i + h * w, j + n
            first, _ = label(b, structure=np.ones((3, 3)))
            labels, count = morphological_components(b)
            assert count == n and np.array_equal(labels[b], want[first[b] - 1])
            assert not np.array_equal(labels, first)
        assert len(data["shapes"]) == 146 and j == data["perm"].size

    @staticmethod
    def _pattern_mosaic(h, w, cols):
        # every h x w binary pattern (itertools.product order), row-major in cells of
        # (h + 1) x (w + 1) px, cols cells per row: the blank separators keep them apart
        pats = np.array(list(itertools.product((0, 1), repeat=h * w)), dtype=bool).reshape(-1, h, w)
        rows = -(-len(pats) // cols)
        out = np.zeros((rows * (h + 1), cols * (w + 1)), dtype=bool)
        for k, p in enumerate(pats):
            i, j = divmod(k, cols)
            out[i * (h + 1):i * (h + 1) + h, j * (w + 1):j * (w + 1) + w] = p
        return out

    @pytest.mark.parametrize(
        "h, w, cols, n, sha256",
        [
            (3, 4, 64, 5904, "e3e24d64486f5f6248cbdf86354facecfa4b766e95d2c67d06cc7728a05cbbb5"),
            (4, 3, 64, 5904, "a24c2a36269c59b1fb12dbe37678a72afb6374ceb0bfbcf8bb290724e6292a85"),
            (4, 4, 256, 103696, "9d6e9e87917f2a908b6207c7698fd39b6d965fbd1f9aedbea4f257cb5112cb40"),
            (3, 5, 256, 52496, "75219d6e6d91a3d8a4cf8de229c1948466cf5a8e4e69bc9ab743f6bbc4f6d097"),
            (5, 3, 256, 52496, "ade512bac1c1b77f3d09329cf7db35f0092a95358c1e3b05d3d468f6a8066375"),
        ],
    )
    def test_every_small_pattern_matches_mathematica(self, h, w, cols, n, sha256):
        # SHA-256 of Mathematica 14.3's MorphologicalComponents of the mosaic (int32, little
        # endian); scipy's first-pixel numbering fails every one of these mosaics
        labels, count = morphological_components(self._pattern_mosaic(h, w, cols))
        assert count == n
        assert hashlib.sha256(labels.astype("<i4").tobytes()).hexdigest() == sha256

    def test_partition_is_8_connected_components(self):
        from scipy.ndimage import label

        b = np.random.default_rng(3).random((120, 90)) < 0.45
        labels, n = morphological_components(b)
        ref, n_ref = label(b, structure=np.ones((3, 3)))
        assert n == n_ref
        assert len(set(zip(labels[b].tolist(), ref[b].tolist()))) == n

    def test_empty(self):
        labels, n = morphological_components(np.zeros((3, 4), dtype=bool))
        assert n == 0 and not labels.any()


class TestResizeBinary:
    def test_identity(self):
        b = np.random.default_rng(1).random((17, 23)) > 0.5
        assert np.array_equal(resize_binary(b, (17, 23)), b)

    def test_upsampling_ties_take_both_pixels(self):
        # 32 -> 40: output 2's centre falls exactly between inputs 1 and 2
        b = np.zeros((32, 1), dtype=bool)
        b[1, 0] = True
        out = resize_binary(b, (40, 1))[:, 0]
        assert out[1] and out[2]
        b = np.zeros((32, 1), dtype=bool)
        b[2, 0] = True
        assert resize_binary(b, (40, 1))[2, 0]


class TestPruneSkeleton:
    def test_is_the_exact_mathematica_port(self, simple_skeleton):
        from kymobutler.pruning import mma_pruning

        for n in (2, 3, 5):
            assert np.array_equal(prune_skeleton(simple_skeleton, n), mma_pruning(simple_skeleton, n))


class TestImageSize:
    @pytest.mark.parametrize("shape", [(8, 50), (50, 8), (8, 8), (1, 1), (1, 300)])
    def test_too_small_raises_clear_error(self, shape):
        h, w = shape
        msg = f"KymoButler needs an image of at least 9 x 9 pixels; this image is {w} x {h}"
        with pytest.raises(ValueError, match=msg):
            check_image_size(shape)
        with pytest.raises(ValueError, match=msg):
            process_segmentation_bi(np.zeros((16, 48)), shape)
        with pytest.raises(ValueError, match=msg):
            process_segmentation_uni(np.zeros((16, 48)), shape)

    def test_prediction_already_at_image_size_is_not_a_network_resize(self):
        # the size limit comes from the network resolution 16*Round[d/16]; a prediction that is
        # already at the kymograph's size (e.g. a wavelet segmentation) is not resized
        pred = np.zeros((8, 50))
        pred[1:7, 9:12] = 1.0  # a 3 px wide bar: thinned to column 10, pruned at both ends
        assert np.argwhere(process_segmentation_bi(pred, (8, 50), 0.5, 1, 1)).tolist() == [
            [2, 10], [3, 10], [4, 10], [5, 10]]
        assert not process_segmentation_uni(np.zeros((8, 50)), (8, 50)).any()

    def test_nine_pixels_is_enough(self):
        # Mathematica runs the networks at 16*Round[9/16] = 16 px
        check_image_size((9, 9))
        assert not process_segmentation_bi(np.zeros((16, 16)), (9, 9)).any()
        assert not process_segmentation_uni(np.zeros((16, 16)), (9, 9)).any()
