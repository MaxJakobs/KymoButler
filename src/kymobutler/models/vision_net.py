"""Vision module (decision network) for tracking ambiguity resolution.

Corresponds to VisionModule in NeuralNetworkDefs.wl lines 36-45.
The VisionModule takes a kymograph tile, a binary mask of the current track,
and a binary mask of all structures, then predicts a probability map for
the next coordinate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from kymobutler.models.unet import UNet


class ScaledDropout(nn.Module):
    """Dropout that also scales output by (1-p) to match Mathematica behavior.

    In Mathematica's KymoButler, the dropout output is explicitly multiplied by (1-p)
    after the standard dropout operation. PyTorch's nn.Dropout already scales by 1/(1-p)
    during training, so to match the Mathematica behavior we apply an additional (1-p)^2
    during training and (1-p) during eval.
    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.dropout = nn.Dropout2d(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return self.dropout(x) * (1.0 - self.p)
        else:
            return x * (1.0 - self.p)


class VisionNet(nn.Module):
    """Decision module for tracking: predicts next-coordinate probability map.

    Inputs:
        img: (B, 1, tile_size, tile_size) - grayscale kymograph tile
        bin_mask: (B, 1, tile_size, tile_size) - binary mask of current track
        fullbin_mask: (B, 1, tile_size, tile_size) - binary mask of all structures

    Output:
        (B, 2, tile_size, tile_size) - softmax probability map (foreground/background)

    The bin_mask goes through Dropout(0.05) * 0.95, fullbin through Dropout(0.5) * 0.5.
    Then [img, bin_dropped, fullbin_dropped] are concatenated into 3-channel input for a UNet.
    """

    def __init__(self, tile_size: int = 48, n: int = 64):
        super().__init__()
        self.tile_size = tile_size
        self.bin_dropout = ScaledDropout(0.05)
        self.fullbin_dropout = ScaledDropout(0.5)
        self.unet = UNet(n=n, in_channels=3)

    def forward(
        self,
        img: torch.Tensor,
        bin_mask: torch.Tensor,
        fullbin_mask: torch.Tensor,
    ) -> torch.Tensor:
        bin_d = self.bin_dropout(bin_mask)
        fullbin_d = self.fullbin_dropout(fullbin_mask)
        combined = torch.cat([img, bin_d, fullbin_d], dim=1)  # (B, 3, H, W)
        return self.unet(combined)  # (B, 2, H, W)
