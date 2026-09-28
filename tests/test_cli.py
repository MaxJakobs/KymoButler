"""End-to-end tests of the ``kymobutler analyze`` command on the public test images: the files it
writes must be Mathematica's (tests/data/reference/pproc_mma.json holds Mathematica's summary.csv
and trackCoordinates.csv for the Mathematica tracks of these images)."""

import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from PIL import Image

from kymobutler import __version__
from kymobutler.cli import cli

DATA = Path(__file__).parent / "data"
REF = json.loads((DATA / "reference" / "pproc_mma.json").read_text())
MODELS = Path(__file__).parent.parent / "models"
_ONNX = ("bidirectional_seg.onnx", "unidirectional_seg.onnx", "decision_module.onnx",
         "classifier.onnx")
needs_models = pytest.mark.skipif(
    not all((MODELS / f).exists() for f in _ONNX), reason="model weights not available"
)


def _analyze(tmp_path, image, *args):
    res = CliRunner().invoke(
        cli, ["analyze", str(DATA / image), "--model-dir", str(MODELS), "-o", str(tmp_path), *args],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output
    return res


@needs_models
@pytest.mark.parametrize(
    "image,args,key",
    [
        ("bitest.png", ["--mode", "bidirectional", "--pixel-time", "0.5", "--pixel-space", "0.1"],
         "bi_bitest_t0.2_s10_f10|0.5|0.1"),
        ("unitest2.png", ["--mode", "unidirectional", "--min-size", "3", "--min-frames", "3",
                          "--pixel-time", "0.3", "--pixel-space", "0.7"],
         "uni_unitest2_t20_s3_f3|0.3|0.7"),
    ],
)
def test_analyze_writes_mathematicas_files(tmp_path, image, args, key):
    _analyze(tmp_path, image, *args)
    stem = Path(image).stem
    summary = (tmp_path / f"{stem}_statistics.csv").read_text()
    want = REF["csv"][key]
    header, rows = summary.split("\n", 1)
    assert rows == want["summary"].split("\n", 1)[1]  # column names and per-track rows
    ts, xs = key.split("|")[1:]
    min_frames = 10 if "bidirectional" in args else 3
    assert header == ",".join(f'"{c}"' for c in (
        f"KymoButler Version {__version__} Summary", f"pixelsize time= {ts} sec",
        f"pixelsize space= {xs} um", f"minimum frames= {min_frames}",
        f"minimum obj size= {min_frames}", "threshold= 0.2"))
    assert (tmp_path / f"{stem}_tracks.csv").read_text() == want["coords"]
    overlay = np.asarray(Image.open(tmp_path / f"{stem}_overlay.png"))
    assert overlay.shape[2] == 4 and (overlay[..., 3] == 255).all()


@needs_models
def test_analyze_json_without_overlay(tmp_path):
    _analyze(tmp_path, "unitest2.png", "--mode", "unidirectional", "--min-size", "3",
             "--min-frames", "3", "--output-format", "json", "--no-overlay")
    tracks = json.loads((tmp_path / "unitest2_tracks.json").read_text())["tracks"]
    want = REF["sets"]["uni_unitest2_t20_s3_f3"]  # 1-based, anterograde then retrograde
    assert [[[p["time"] + 1, p["space"] + 1] for p in t["points"]] for t in tracks] == want
    assert not (tmp_path / "unitest2_overlay.png").exists()


@pytest.mark.parametrize(
    "args",
    [["--min-frames", "0"], ["--pixel-time", "0"], ["--pixel-space", "-1"], ["--threshold", "1.5"]],
)
def test_analyze_rejects_invalid_parameters(tmp_path, args):
    res = CliRunner().invoke(
        cli, ["analyze", str(DATA / "unitest2.png"), "-o", str(tmp_path), *args]
    )
    assert res.exit_code == 2 and "Invalid value" in res.output


@needs_models
def test_analyze_wavelet_uses_the_decision_module(tmp_path, monkeypatch):
    import kymobutler.wavelet as wavelet

    seen = {}
    real = wavelet.analyze_wavelet_bidirectional

    def spy(image, threshold, vision_net, *args):
        seen["vision_net"] = vision_net
        return real(image, threshold, vision_net, *args)

    monkeypatch.setattr(wavelet, "analyze_wavelet_bidirectional", spy)
    _analyze(tmp_path, "unitest2.png", "--mode", "wavelet", "--min-size", "3", "--min-frames", "3")
    assert seen["vision_net"] is not None  # AnalyseKymographBIwavelet tracks with vismod
    header = (tmp_path / "unitest2_statistics.csv").read_text().splitlines()[1]
    assert header.startswith('"Direction","Av frame2frame velocity [um/sec]"')


def test_analyze_without_models_is_a_clear_error(tmp_path):
    image = str(DATA / "unitest2.png")
    res = CliRunner().invoke(
        cli, ["analyze", image, "--model-dir", str(tmp_path), "-o", str(tmp_path)]
    )
    assert res.exit_code == 1 and "Error: Model files not found" in res.output
    assert not list(tmp_path.glob("unitest2_*"))


def test_download_models_reports_the_files_used(tmp_path):
    res = CliRunner().invoke(cli, ["download-models", "--model-dir", str(tmp_path)])
    assert res.exit_code == 1 and res.output.count("[MISSING]") == 4
    if (MODELS / "bidirectional_seg.onnx").exists():
        res = CliRunner().invoke(cli, ["download-models", "--model-dir", str(MODELS)])
        assert res.exit_code == 0 and "bidirectional_seg.onnx [OK]" in res.output
