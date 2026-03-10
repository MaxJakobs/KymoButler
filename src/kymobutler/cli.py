"""Command-line interface for KymoButler."""

from __future__ import annotations

from pathlib import Path

import click


@click.group()
@click.version_option(version="2.0.0")
def cli():
    """KymoButler: AI-powered kymograph analysis."""


@cli.command()
@click.argument("image", type=click.Path(exists=True))
@click.option(
    "--mode",
    type=click.Choice(["bidirectional", "unidirectional", "wavelet"]),
    default="bidirectional",
    help="Analysis mode.",
)
@click.option("--threshold", "-t", type=float, default=0.2, help="Segmentation threshold.")
@click.option("--vision-threshold", "-v", type=float, default=0.5, help="Vision module threshold.")
@click.option("--min-size", type=int, default=10, help="Minimum track size in pixels.")
@click.option("--min-frames", type=int, default=10, help="Minimum track duration in frames.")
@click.option("--pixel-time", type=float, default=1.0, help="Time pixel size (seconds).")
@click.option("--pixel-space", type=float, default=1.0, help="Space pixel size (micrometers).")
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
    help="Track data output format.",
)
@click.option("--save-overlay/--no-overlay", default=True, help="Save overlay visualization.")
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
):
    """Analyze a kymograph image."""
    import torch

    from kymobutler.io_utils import create_overlay, save_statistics_csv, save_tracks_csv, save_tracks_json
    from kymobutler.models.weights import load_default_models
    from kymobutler.postprocessing import postprocess
    from kymobutler.segmentation import segment_bidirectional, segment_unidirectional
    from kymobutler.tracking import track_bidirectional, track_unidirectional
    from kymobutler.wavelet import analyze_wavelet_bidirectional

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    image_name = Path(image).stem

    if mode == "wavelet":
        click.echo("Analyzing with wavelet segmentation...")
        # Wavelet mode can optionally use the vision net
        vision_net = None
        if model_dir or Path.home().joinpath(".kymobutler/models/decision_module.pt").exists():
            try:
                models = load_default_models(model_dir, device)
                vision_net = models["decnet"]
            except FileNotFoundError:
                click.echo("Vision module not found, using wavelet-only tracking.")

        tracks = analyze_wavelet_bidirectional(
            image, threshold, vision_net, vision_threshold, min_size, min_frames, device
        )
    else:
        # Load neural network models
        click.echo("Loading models...")
        models = load_default_models(model_dir, device)

        if mode == "bidirectional":
            click.echo("Segmenting (bidirectional)...")
            was_negated, raw, preprocessed, pred = segment_bidirectional(
                image, models["binet"], device
            )
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
            click.echo("Tracking...")
            ant_tracks, ret_tracks = track_unidirectional(
                pred_dict, preprocessed.shape, threshold, min_size, min_frames
            )
            tracks = ant_tracks + ret_tracks

    click.echo(f"Found {len(tracks)} tracks.")

    # Save tracks
    track_file = output_path / f"{image_name}_tracks.{output_format}"
    if output_format == "csv":
        save_tracks_csv(tracks, track_file)
    else:
        save_tracks_json(tracks, track_file)
    click.echo(f"Tracks saved to {track_file}")

    # Post-process and save statistics
    stats = postprocess(tracks, pixel_time, pixel_space)
    stats_file = output_path / f"{image_name}_statistics.csv"
    save_statistics_csv(stats, stats_file, pixel_time, pixel_space)
    click.echo(f"Statistics saved to {stats_file}")

    # Save overlay
    if save_overlay:
        from kymobutler.preprocessing import load_and_preprocess

        preprocessed, raw, _ = load_and_preprocess(image)
        overlay_file = output_path / f"{image_name}_overlay.png"
        create_overlay(raw, tracks, overlay_file)
        click.echo(f"Overlay saved to {overlay_file}")


@cli.command("download-models")
@click.option("--model-dir", type=click.Path(), default=None, help="Target directory.")
def download_models(model_dir):
    """Check if model weights are available."""
    from kymobutler.config import DEFAULT_MODEL_DIR, WEIGHT_FILES

    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
    click.echo(f"Model directory: {model_dir}")

    all_present = True
    for key, filename in WEIGHT_FILES.items():
        path = model_dir / filename
        status = "OK" if path.exists() else "MISSING"
        if status == "MISSING":
            all_present = False
        click.echo(f"  {key}: {path} [{status}]")

    if not all_present:
        click.echo(
            "\nTo obtain model weights, run the ONNX conversion pipeline:\n"
            "  1. In Mathematica: wolframscript -file scripts/export_to_onnx.wls\n"
            "  2. In Python: python scripts/convert_weights.py"
        )


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
