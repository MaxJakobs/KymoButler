"""Validate converted PyTorch models against Mathematica reference outputs.

Prerequisites:
    - Run scripts/export_to_onnx.wls (generates reference outputs)
    - Run scripts/convert_weights.py (generates PyTorch weights)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from kymobutler.models.unet import UNet
from kymobutler.preprocessing import load_and_preprocess, resize_to_multiple_of_16


def validate(
    model_path: Path,
    test_image_path: Path,
    reference_path: Path,
    tolerance: float = 1e-4,
) -> bool:
    """Compare PyTorch model output against Mathematica reference."""
    print(f"Validating {model_path.name}...")

    model = UNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    preprocessed, _, _ = load_and_preprocess(test_image_path)
    resized = resize_to_multiple_of_16(preprocessed)
    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float()

    with torch.no_grad():
        pred = model(tensor).numpy()

    expected = np.load(reference_path)

    max_diff = np.max(np.abs(pred - expected))
    print(f"  Max absolute difference: {max_diff:.6f}")

    if max_diff < tolerance:
        print(f"  PASS (tolerance: {tolerance})")
        return True
    else:
        print(f"  FAIL (tolerance: {tolerance})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate converted models")
    parser.add_argument("--model-dir", type=Path, default=Path.home() / ".kymobutler" / "models")
    parser.add_argument("--reference-dir", type=Path, default=Path("models"))
    parser.add_argument("--test-image", type=Path, default=Path("TestAndDeploy/bitest.png"))
    args = parser.parse_args()

    model_path = args.model_dir / "bidirectional_seg.pt"
    ref_path = args.reference_dir / "bitest_reference_output.npy"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run scripts/convert_weights.py first.")
        return

    if not ref_path.exists():
        print(f"Reference output not found: {ref_path}")
        print("Run scripts/export_to_onnx.wls first.")
        return

    validate(model_path, args.test_image, ref_path)


if __name__ == "__main__":
    main()
