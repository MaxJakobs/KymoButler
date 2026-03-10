"""Model weight download and cache management."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from kymobutler.config import DEFAULT_MODEL_DIR, WEIGHT_FILES


class OnnxBiNet(nn.Module):
    """Wrapper around ONNX-converted bidirectional segmentation model.

    The ONNX model outputs (B, H, W) after internal softmax+argmax/slice.
    We wrap it to match the expected (B, 2, H, W) interface by adding a
    background channel.
    """

    def __init__(self, onnx_model: nn.Module):
        super().__init__()
        self.model = onnx_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)  # (B, H, W) - foreground probability
        if out.dim() == 3:
            bg = 1.0 - out
            return torch.stack([out, bg], dim=1)  # (B, 2, H, W)
        return out


class OnnxUniNet(nn.Module):
    """Wrapper around ONNX-converted unidirectional segmentation model.

    The ONNX model returns a list [ant, ret], each (B, H, W).
    """

    def __init__(self, onnx_model: nn.Module):
        super().__init__()
        self.model = onnx_model

    def forward(self, x: torch.Tensor) -> dict:
        out = self.model(x)  # list of 2 tensors, each (B, H, W)
        ant = out[0] if isinstance(out, list) else out
        ret = out[1] if isinstance(out, list) and len(out) > 1 else torch.zeros_like(ant)
        # Wrap into 2-channel format
        if ant.dim() == 3:
            ant = torch.stack([ant, 1.0 - ant], dim=1)
        if ret.dim() == 3:
            ret = torch.stack([ret, 1.0 - ret], dim=1)
        return {"ant": ant, "ret": ret}


class OnnxDecNet(nn.Module):
    """Wrapper around ONNX-converted decision module.

    The ONNX model expects (B, 3, 48, 48) and outputs (B, 48, 48, 2).
    We need it to accept separate inputs and output (B, 2, 48, 48).
    """

    def __init__(self, onnx_model: nn.Module):
        super().__init__()
        self.model = onnx_model

    def forward(
        self,
        img: torch.Tensor,
        bin_mask: torch.Tensor,
        fullbin_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Concatenate inputs as the ONNX model expects (B, 3, H, W)
        combined = torch.cat([img, bin_mask, fullbin_mask], dim=1)
        out = self.model(combined)  # (B, H, W, 2)
        if out.dim() == 4 and out.shape[-1] == 2:
            out = out.permute(0, 3, 1, 2)  # -> (B, 2, H, W)
        return out


class OnnxClassNet(nn.Module):
    """Wrapper around ONNX-converted classifier.

    The ONNX model may require a TrainingMode flag input.
    """

    def __init__(self, onnx_model: nn.Module):
        super().__init__()
        self.model = onnx_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            return self.model(x)
        except TypeError:
            return self.model(x, torch.tensor(False))


def load_default_models(
    model_dir: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, nn.Module]:
    """Load pre-trained KymoButler models from disk.

    Supports two formats:
    1. ONNX files (.onnx) - converted from Mathematica via export_to_onnx.wls
    2. PyTorch state dicts (.pt) - from our native architecture definitions

    Args:
        model_dir: Directory containing model files. Defaults to ~/.kymobutler/models/
        device: Device to load models onto ('cpu', 'cuda', 'mps').

    Returns:
        Dictionary with keys 'binet', 'uninet', 'decnet', 'classnet'.
    """
    model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR

    # ONNX file paths (from Mathematica export)
    onnx_files = {
        "binet": "bidirectional_seg.onnx",
        "uninet": "unidirectional_seg.onnx",
        "decnet": "decision_module.onnx",
        "classnet": "classifier.onnx",
    }

    # Try ONNX files first (from Mathematica export), then fall back to .pt files
    models: dict[str, nn.Module] = {}
    missing = []

    for key in ("binet", "uninet", "decnet", "classnet"):
        onnx_path = model_dir / onnx_files[key]
        pt_path = model_dir / WEIGHT_FILES[key]

        if onnx_path.exists():
            models[key] = _load_onnx_model(key, onnx_path, device)
        elif pt_path.exists():
            models[key] = _load_pt_model(key, pt_path, device)
        else:
            missing.append(f"{onnx_path} or {pt_path}")

    if missing:
        raise FileNotFoundError(
            f"Model files not found: {', '.join(missing)}. "
            "Run the ONNX export pipeline:\n"
            "  1. wolframscript -file scripts/export_to_onnx.wls\n"
            "  2. Copy .onnx files to " + str(model_dir)
        )

    return models


def _load_onnx_model(key: str, path: Path, device: str) -> nn.Module:
    """Load an ONNX model and wrap it with the appropriate adapter."""
    import onnx
    from onnx2torch import convert

    onnx_model = onnx.load(str(path))
    torch_model = convert(onnx_model)
    torch_model.to(device)
    torch_model.eval()

    wrappers = {
        "binet": OnnxBiNet,
        "uninet": OnnxUniNet,
        "decnet": OnnxDecNet,
        "classnet": OnnxClassNet,
    }
    wrapped = wrappers[key](torch_model)
    wrapped.to(device)
    wrapped.eval()
    return wrapped


def _load_pt_model(key: str, path: Path, device: str) -> nn.Module:
    """Load a native PyTorch model from a state dict .pt file."""
    from kymobutler.models.unet import UNet, UNetUnidirectional
    from kymobutler.models.vision_net import VisionNet
    from kymobutler.models.classnet import ClassNet

    constructors = {
        "binet": lambda: UNet(),
        "uninet": lambda: UNetUnidirectional(),
        "decnet": lambda: VisionNet(),
        "classnet": lambda: ClassNet(n_classes=2, input_size=48),
    }

    model = constructors[key]()
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
