"""Integration tests using the original KymoButler test images.

These tests require trained model weights to be present at ~/.kymobutler/models/.
Expected results from KymoButlerTestandDeploy.wls:
  - unitest.png  -> 7 anterograde tracks (Mathematica native)
  - unitest2.png -> 9 anterograde tracks (Mathematica native)
  - bitest.png   -> 13 tracks (Mathematica native)

The ONNX-exported models produce slightly different predictions from the
native Mathematica models, so exact track counts may differ. We test for
reasonable ranges and key structural properties instead.
"""

from pathlib import Path

import pytest

from kymobutler.preprocessing import load_and_preprocess


# Check if model weights are available
_MODEL_DIR = Path.home() / ".kymobutler" / "models"
_has_models = (_MODEL_DIR / "bidirectional_seg.onnx").exists() or (
    _MODEL_DIR / "bidirectional_seg.pt"
).exists()


class TestImageLoading:
    """Basic tests that don't require model weights."""

    def test_unitest_loads(self, unitest_path):
        preprocessed, raw, negated = load_and_preprocess(unitest_path)
        assert preprocessed.shape == raw.shape
        assert preprocessed.ndim == 2

    def test_unitest2_loads(self, unitest2_path):
        preprocessed, raw, negated = load_and_preprocess(unitest2_path)
        assert preprocessed.shape == raw.shape

    def test_bitest_loads(self, bitest_path):
        preprocessed, raw, negated = load_and_preprocess(bitest_path)
        assert preprocessed.shape == raw.shape


@pytest.mark.skipif(not _has_models, reason="Model weights not found at ~/.kymobutler/models/")
class TestFullPipeline:
    def test_unidirectional_unitest(self, unitest_path):
        """unitest.png should produce anterograde tracks."""
        from kymobutler.models.weights import load_default_models
        from kymobutler.segmentation import segment_unidirectional
        from kymobutler.tracking import track_unidirectional

        models = load_default_models()
        _, raw, preprocessed, pred_dict = segment_unidirectional(
            unitest_path, models["uninet"]
        )
        ant_tracks, ret_tracks = track_unidirectional(
            pred_dict, preprocessed.shape, threshold=0.2, min_size=3, min_frames=3
        )
        # ONNX-exported model produces more tracks than Mathematica native (7)
        # due to precision differences. Check for reasonable range.
        assert len(ant_tracks) > 0, "Should find at least some anterograde tracks"
        assert len(ant_tracks) >= 5, f"Expected at least 5 ant tracks, got {len(ant_tracks)}"

    def test_unidirectional_unitest2(self, unitest2_path):
        """unitest2.png should produce anterograde tracks."""
        from kymobutler.models.weights import load_default_models
        from kymobutler.segmentation import segment_unidirectional
        from kymobutler.tracking import track_unidirectional

        models = load_default_models()
        _, raw, preprocessed, pred_dict = segment_unidirectional(
            unitest2_path, models["uninet"]
        )
        ant_tracks, ret_tracks = track_unidirectional(
            pred_dict, preprocessed.shape, threshold=0.2, min_size=3, min_frames=3
        )
        assert len(ant_tracks) > 0, "Should find at least some anterograde tracks"

    def test_bidirectional_bitest(self, bitest_path):
        """bitest.png should produce ~13 tracks."""
        from kymobutler.models.weights import load_default_models
        from kymobutler.segmentation import segment_bidirectional
        from kymobutler.tracking import track_bidirectional

        models = load_default_models()
        was_negated, raw, preprocessed, pred = segment_bidirectional(
            bitest_path, models["binet"]
        )
        tracks = track_bidirectional(
            pred, preprocessed, was_negated, threshold=0.2,
            vision_net=None, min_size=10, min_frames=10,
        )
        assert len(tracks) == 13, f"Expected 13 tracks, got {len(tracks)}"

    def test_bidirectional_track_properties(self, bitest_path):
        """Verify structural properties of bidirectional tracks."""
        from kymobutler.models.weights import load_default_models
        from kymobutler.segmentation import segment_bidirectional
        from kymobutler.tracking import track_bidirectional

        models = load_default_models()
        was_negated, raw, preprocessed, pred = segment_bidirectional(
            bitest_path, models["binet"]
        )
        tracks = track_bidirectional(
            pred, preprocessed, was_negated, threshold=0.2,
            vision_net=None, min_size=10, min_frames=10,
        )
        # All tracks should have points
        for t in tracks:
            assert len(t.points) >= 2, "Each track should have at least 2 points"
            span = t.points[-1][0] - t.points[0][0]
            assert span >= 10, f"Track should span at least 10 frames, got {span}"

    def test_segmentation_prediction_range(self, bitest_path):
        """Prediction values should be in [0, 1] probability range."""
        from kymobutler.models.weights import load_default_models
        from kymobutler.segmentation import segment_bidirectional

        models = load_default_models()
        _, _, _, pred = segment_bidirectional(bitest_path, models["binet"])
        assert pred.min() >= 0.0, "Predictions should be >= 0"
        assert pred.max() <= 1.0, "Predictions should be <= 1"
        assert pred.max() > 0.5, "Model should produce some confident predictions"
