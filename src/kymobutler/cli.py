"""Command-line interface for KymoButler."""

from __future__ import annotations

from pathlib import Path

import click

from kymobutler import __version__


@click.group()
@click.version_option(version=__version__)
def cli():
    """KymoButler: AI-powered kymograph analysis."""


_POSITIVE = click.FloatRange(min=0, min_open=True)


@cli.command()
@click.argument("image", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--mode",
    type=click.Choice(["bidirectional", "unidirectional", "wavelet"]),
    default="bidirectional",
    help="Analysis mode.",
)
@click.option(
    "--threshold", "-t", type=click.FloatRange(0, 1), default=0.2, help="Segmentation threshold."
)
@click.option(
    "--vision-threshold", "-v", type=click.FloatRange(0, 1), default=0.5,
    help="Vision module threshold.",
)
@click.option(
    "--min-size", type=click.IntRange(min=1), default=10, help="Minimum track size in pixels."
)
@click.option(
    "--min-frames",
    type=click.IntRange(min=1),
    default=10,
    help="Minimum track duration in frames (at least 1: a one-point track has no velocity).",
)
@click.option(
    "--pixel-time", type=_POSITIVE, default=1.0, help="Time pixel size (seconds per frame)."
)
@click.option("--pixel-space", type=_POSITIVE, default=1.0, help="Space pixel size (micrometers).")
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cpu",
    help="Computation device.",
)
@click.option("--model-dir", type=click.Path(), default=None, help="Model weights directory.")
@click.option("--output-dir", "-o", type=click.Path(), default=".", help="Output directory.")
@click.option(
    "--output-format",
    type=click.Choice(["csv", "json"]),
    default="csv",
    help="Track data output format: csv as KymoButler's trackCoordinates.csv, or json.",
)
@click.option("--save-overlay/--no-overlay", default=True, help="Save overlay visualization.")
@click.option(
    "--save-debug/--no-debug", default=False,
    help="Save raw segmentation predictions for debugging.",
)
def analyze(
    image,
    mode,
    threshold,
    vision_threshold,
    min_size,
    min_frames,
    pixel_time,
    pixel_space,
    device,
    model_dir,
    output_dir,
    output_format,
    save_overlay,
    save_debug,
):
    """Analyze a kymograph image.

    Writes <image>_tracks.csv (or .json), <image>_statistics.csv (KymoButler's summary.csv)
    and <image>_overlay.png into the output directory.
    """
    from kymobutler.io_utils import (
        create_overlay,
        save_segmentation_debug,
        save_statistics_csv,
        save_tracks_csv,
        save_tracks_json,
    )
    from kymobutler.models.weights import load_default_models
    from kymobutler.postprocessing import postprocess
    from kymobutler.segmentation import segment_bidirectional, segment_unidirectional
    from kymobutler.tracking import track_bidirectional, track_unidirectional
    from kymobutler.wavelet import analyze_wavelet_bidirectional

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_name = Path(image).stem

    # Load the neural network models (the wavelet mode tracks with the decision module too, as
    # WaveletPackage.wl's AnalyseKymographBIwavelet does); a missing model is an error.
    click.echo("Loading models...")
    try:
        models = load_default_models(model_dir, device)
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from None

    if mode == "wavelet":
        click.echo("Analyzing with wavelet segmentation...")
        tracks = analyze_wavelet_bidirectional(
            image, threshold, models["decnet"], vision_threshold, min_size, min_frames, device
        )
        raw = None
    elif mode == "bidirectional":
        click.echo("Segmenting (bidirectional)...")
        was_negated, raw, preprocessed, pred = segment_bidirectional(image, models["binet"], device)
        if save_debug:
            debug_dir = output_path / f"{image_name}_debug"
            save_segmentation_debug(pred, debug_dir, "binet")
            click.echo(f"Debug predictions saved to {debug_dir}")
        click.echo("Tracking...")
        tracks = track_bidirectional(
            pred, preprocessed, was_negated, threshold, vision_threshold,
            models["decnet"], min_size, min_frames, device,
        )
    else:
        click.echo("Segmenting (unidirectional)...")
        was_negated, raw, preprocessed, pred_dict = segment_unidirectional(
            image, models["uninet"], device
        )
        if save_debug:
            debug_dir = output_path / f"{image_name}_debug"
            save_segmentation_debug(pred_dict, debug_dir, "uninet")
            click.echo(f"Debug predictions saved to {debug_dir}")
        click.echo("Tracking...")
        ant_tracks, ret_tracks = track_unidirectional(
            pred_dict, preprocessed.shape, threshold, min_size, min_frames
        )
        tracks = ant_tracks + ret_tracks  # numbered as Mathematica does: ant, then ret

    click.echo(f"Found {len(tracks)} tracks.")
    # Statistics first: a track without defined statistics stops the run before any file is written.
    try:
        stats = postprocess(tracks, pixel_time, pixel_space)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None

    # Save tracks
    track_file = output_path / f"{image_name}_tracks.{output_format}"
    if output_format == "csv":
        save_tracks_csv(tracks, track_file, pixel_time, pixel_space)
    else:
        save_tracks_json(tracks, track_file)
    click.echo(f"Tracks saved to {track_file}")

    # Save statistics (KymoButler's summary.csv)
    stats_file = output_path / f"{image_name}_statistics.csv"
    save_statistics_csv(
        stats, stats_file, pixel_time, pixel_space,
        min_frames=min_frames, min_size=min_size, threshold=threshold,
    )
    click.echo(f"Statistics saved to {stats_file}")

    # Save overlay
    if save_overlay:
        if raw is None:  # wavelet mode: the same rawkym the segmentation used
            from kymobutler.preprocessing import load_and_preprocess

            _, raw, _ = load_and_preprocess(image)
        overlay_file = output_path / f"{image_name}_overlay.png"
        create_overlay(raw, tracks, overlay_file)
        click.echo(f"Overlay saved to {overlay_file}")


# The model files load_default_models looks for, in its order: the ONNX export, else the
# PyTorch state dict made from it by scripts/convert_weights.py.
_ONNX_FILES = {
    "binet": "bidirectional_seg.onnx",
    "uninet": "unidirectional_seg.onnx",
    "decnet": "decision_module.onnx",
    "classnet": "classifier.onnx",
}


@cli.command("download-models")
@click.option("--model-dir", type=click.Path(), default=None, help="Model weights directory.")
def download_models(model_dir):
    """Check which model weights are available (and which file each model loads from)."""
    from kymobutler.config import DEFAULT_MODEL_DIR, WEIGHT_FILES

    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    click.echo(f"Model directory: {model_dir}")

    all_present = True
    for key, onnx_name in _ONNX_FILES.items():
        candidates = [model_dir / onnx_name, model_dir / WEIGHT_FILES[key]]
        found = next((f for f in candidates if f.exists()), None)
        all_present &= found is not None
        click.echo(f"  {key}: {found or candidates[0]} [{'OK' if found else 'MISSING'}]")

    if not all_present:
        click.echo(
            "\nTo obtain model weights, export them from the Mathematica networks:\n"
            "  1. In Mathematica: wolframscript -file legacy/scripts/export_to_onnx.wls\n"
            "  2. Optionally, for PyTorch state dicts: python scripts/convert_weights.py"
        )
        raise SystemExit(1)


@cli.command()
@click.argument("prediction_tracks", type=click.Path(exists=True))
@click.argument("ground_truth_tracks", type=click.Path(exists=True))
def benchmark(prediction_tracks, ground_truth_tracks):
    """Compute precision/recall/F1 between predicted and ground-truth track files."""
    import json

    from kymobutler.benchmarking import benchmark_prediction
    from kymobutler.tracking import Track

    def load_tracks(path: str) -> list[Track]:
        with open(path) as f:
            data = json.load(f)
        return [
            Track(points=[(p["time"], p["space"]) for p in t["points"]])
            for t in data["tracks"]
        ]

    pred = load_tracks(prediction_tracks)
    gt = load_tracks(ground_truth_tracks)

    results = benchmark_prediction(pred, gt)
    click.echo(f"Recall:    {results['recall']:.4f}")
    click.echo(f"Precision: {results['precision']:.4f}")
    click.echo(f"F1:        {results['f1']:.4f}")
