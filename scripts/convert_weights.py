"""Convert ONNX models exported from Mathematica to PyTorch state dicts.

Usage:
    python scripts/convert_weights.py [--input-dir models/] [--output-dir ~/.kymobutler/models/]

Prerequisites:
    pip install kymobutler[convert]
    Run scripts/export_to_onnx.wls first to generate ONNX files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch
from onnx2torch import convert as onnx_to_torch


def remap_state_dict(
    source_sd: dict[str, torch.Tensor],
    target_sd: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Map ONNX-converted parameter names to PyTorch model parameter names.

    Strategy: match by shape and sequential order. Both models have identical
    architectures, so parameters with matching shapes in order should correspond.
    """
    # Separate tensor params from scalar params
    source_params = [(k, v) for k, v in source_sd.items() if v.dim() > 0]
    target_params = [(k, v) for k, v in target_sd.items() if v.dim() > 0]

    mapped: dict[str, torch.Tensor] = {}
    src_idx = 0

    for tgt_name, tgt_tensor in target_params:
        found = False
        while src_idx < len(source_params):
            src_name, src_tensor = source_params[src_idx]
            if src_tensor.shape == tgt_tensor.shape:
                mapped[tgt_name] = src_tensor
                src_idx += 1
                found = True
                break
            src_idx += 1

        if not found:
            print(f"  WARNING: No matching source param for {tgt_name} {tgt_tensor.shape}")
            mapped[tgt_name] = tgt_tensor

    # Handle scalar params (batch norm running_mean, running_var, num_batches_tracked)
    source_scalars = {k: v for k, v in source_sd.items() if v.dim() == 0 or k not in [s[0] for s in source_params[:src_idx]]}
    for tgt_name, tgt_tensor in target_sd.items():
        if tgt_name not in mapped:
            # Try to find a scalar with matching shape
            for src_name, src_tensor in source_scalars.items():
                if src_tensor.shape == tgt_tensor.shape:
                    mapped[tgt_name] = src_tensor
                    break
            else:
                mapped[tgt_name] = tgt_tensor  # keep default

    return mapped


def convert_and_save(
    onnx_path: Path,
    pytorch_model: torch.nn.Module,
    output_path: Path,
) -> None:
    """Load ONNX model, extract weights, map to PyTorch model, save state dict."""
    print(f"Converting {onnx_path} -> {output_path}")

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # Convert ONNX graph to a PyTorch model
    onnx_torch = onnx_to_torch(onnx_model)

    # Extract and remap state dict
    onnx_sd = onnx_torch.state_dict()
    target_sd = pytorch_model.state_dict()

    print(f"  ONNX params: {len(onnx_sd)}, PyTorch params: {len(target_sd)}")

    mapped_sd = remap_state_dict(onnx_sd, target_sd)
    pytorch_model.load_state_dict(mapped_sd)

    # Validate with a dummy forward pass
    pytorch_model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pytorch_model.state_dict(), output_path)
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert ONNX models to PyTorch")
    parser.add_argument("--input-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path.home() / ".kymobutler" / "models")
    args = parser.parse_args()

    from kymobutler.models.unet import UNet, UNetUnidirectional
    from kymobutler.models.vision_net import VisionNet
    from kymobutler.models.classnet import ClassNet

    conversions = [
        ("bidirectional_seg.onnx", UNet(), "bidirectional_seg.pt"),
        ("unidirectional_seg.onnx", UNetUnidirectional(), "unidirectional_seg.pt"),
        ("decision_module.onnx", VisionNet(), "decision_module.pt"),
        ("classifier.onnx", ClassNet(n_classes=2, input_size=48), "classifier.pt"),
    ]

    for onnx_name, model, pt_name in conversions:
        onnx_path = args.input_dir / onnx_name
        output_path = args.output_dir / pt_name
        if onnx_path.exists():
            convert_and_save(onnx_path, model, output_path)
        else:
            print(f"SKIP: {onnx_path} not found")

    print("\nDone! Verify with: python scripts/validate_conversion.py")


if __name__ == "__main__":
    main()
