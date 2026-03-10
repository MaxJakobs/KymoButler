"""UNet architectures for kymograph segmentation.

Translated from NeuralNetworkDefs.wl (Mathematica).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """Conv2d -> BatchNorm2d -> LeakyReLU(0.1).

    Corresponds to basicBlock[channels, kernelSize] in NeuralNetworkDefs.wl line 21.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DSConvBlock(nn.Module):
    """Depthwise-separable convolution block.

    Depthwise Conv2d(groups=in) -> Pointwise Conv2d(1x1) -> BN -> LeakyReLU.
    Corresponds to dsconvBlock in NeuralNetworkDefs.wl line 28.
    """

    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size,
                stride=stride, padding=1, groups=in_channels,
            ),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Bidirectional segmentation UNet.

    4-level encoder-decoder with skip connections.
    Input: (B, in_channels, H, W) grayscale.
    Output: (B, 2, H, W) softmax probabilities (foreground/background).

    Corresponds to UNET in NeuralNetworkDefs.wl lines 65-94.
    """

    def __init__(self, n: int = 64, in_channels: int = 1):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            BasicBlock(in_channels, n), BasicBlock(n, n), nn.Dropout2d(0.1)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(
            BasicBlock(n, 2 * n), BasicBlock(2 * n, 2 * n), nn.Dropout2d(0.1)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(
            BasicBlock(2 * n, 4 * n), BasicBlock(4 * n, 4 * n), nn.Dropout2d(0.1)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = nn.Sequential(
            BasicBlock(4 * n, 8 * n), BasicBlock(8 * n, 8 * n), nn.Dropout2d(0.1)
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            BasicBlock(8 * n, 16 * n), BasicBlock(16 * n, 16 * n), nn.Dropout2d(0.2)
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(16 * n, 8 * n, 2, stride=2)
        self.dec1 = nn.Sequential(BasicBlock(16 * n, 8 * n), BasicBlock(8 * n, 8 * n))
        self.up2 = nn.ConvTranspose2d(8 * n, 4 * n, 2, stride=2)
        self.dec2 = nn.Sequential(BasicBlock(8 * n, 4 * n), BasicBlock(4 * n, 4 * n))
        self.up3 = nn.ConvTranspose2d(4 * n, 2 * n, 2, stride=2)
        self.dec3 = nn.Sequential(BasicBlock(4 * n, 2 * n), BasicBlock(2 * n, 2 * n))
        self.up4 = nn.ConvTranspose2d(2 * n, n, 2, stride=2)
        self.dec4 = nn.Sequential(BasicBlock(2 * n, n), BasicBlock(n, n))

        # Output head: 1x1 conv to 2 classes
        self.head = nn.Conv2d(n, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d1 = self.dec1(torch.cat([self.up1(b), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))

        return torch.softmax(self.head(d4), dim=1)


class UNetUnidirectional(nn.Module):
    """Two-headed UNet for unidirectional kymograph analysis.

    Same encoder/decoder as UNet but with separate anterograde and retrograde output heads.
    Output: dict with 'ant' and 'ret' keys, each (B, 2, H, W) softmax probabilities.

    Corresponds to UNETunidirectional in NeuralNetworkDefs.wl lines 96-127.
    """

    def __init__(self, n: int = 64, in_channels: int = 1):
        super().__init__()
        # Encoder (same as UNet)
        self.enc1 = nn.Sequential(
            BasicBlock(in_channels, n), BasicBlock(n, n), nn.Dropout2d(0.1)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(
            BasicBlock(n, 2 * n), BasicBlock(2 * n, 2 * n), nn.Dropout2d(0.1)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(
            BasicBlock(2 * n, 4 * n), BasicBlock(4 * n, 4 * n), nn.Dropout2d(0.1)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = nn.Sequential(
            BasicBlock(4 * n, 8 * n), BasicBlock(8 * n, 8 * n), nn.Dropout2d(0.1)
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            BasicBlock(8 * n, 16 * n), BasicBlock(16 * n, 16 * n), nn.Dropout2d(0.2)
        )

        # Decoder (same as UNet)
        self.up1 = nn.ConvTranspose2d(16 * n, 8 * n, 2, stride=2)
        self.dec1 = nn.Sequential(BasicBlock(16 * n, 8 * n), BasicBlock(8 * n, 8 * n))
        self.up2 = nn.ConvTranspose2d(8 * n, 4 * n, 2, stride=2)
        self.dec2 = nn.Sequential(BasicBlock(8 * n, 4 * n), BasicBlock(4 * n, 4 * n))
        self.up3 = nn.ConvTranspose2d(4 * n, 2 * n, 2, stride=2)
        self.dec3 = nn.Sequential(BasicBlock(4 * n, 2 * n), BasicBlock(2 * n, 2 * n))
        self.up4 = nn.ConvTranspose2d(2 * n, n, 2, stride=2)
        self.dec4 = nn.Sequential(BasicBlock(2 * n, n), BasicBlock(n, n))

        # Two output heads
        self.head_ant = nn.Conv2d(n, 2, 1)
        self.head_ret = nn.Conv2d(n, 2, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d1 = self.dec1(torch.cat([self.up1(b), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))

        return {
            "ant": torch.softmax(self.head_ant(d4), dim=1),
            "ret": torch.softmax(self.head_ret(d4), dim=1),
        }


class UNetDSW(nn.Module):
    """Bidirectional UNet with depth-separable convolutions.

    Uses DSConvBlock instead of BasicBlock for efficiency.
    Corresponds to UNETdsw in NeuralNetworkDefs.wl lines 165-194.
    """

    def __init__(self, n: int = 64, in_channels: int = 1):
        super().__init__()
        # Encoder: first block uses BasicBlock, rest use DSConvBlock
        self.enc1 = nn.Sequential(
            BasicBlock(in_channels, n), DSConvBlock(n, n), nn.Dropout2d(0.1)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(
            DSConvBlock(n, 2 * n), DSConvBlock(2 * n, 2 * n), nn.Dropout2d(0.1)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(
            DSConvBlock(2 * n, 4 * n), DSConvBlock(4 * n, 4 * n), nn.Dropout2d(0.1)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = nn.Sequential(
            DSConvBlock(4 * n, 8 * n), DSConvBlock(8 * n, 8 * n), nn.Dropout2d(0.1)
        )
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(
            DSConvBlock(8 * n, 16 * n), DSConvBlock(16 * n, 16 * n), nn.Dropout2d(0.2)
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(16 * n, 8 * n, 2, stride=2)
        self.dec1 = nn.Sequential(DSConvBlock(16 * n, 8 * n), DSConvBlock(8 * n, 8 * n))
        self.up2 = nn.ConvTranspose2d(8 * n, 4 * n, 2, stride=2)
        self.dec2 = nn.Sequential(DSConvBlock(8 * n, 4 * n), DSConvBlock(4 * n, 4 * n))
        self.up3 = nn.ConvTranspose2d(4 * n, 2 * n, 2, stride=2)
        self.dec3 = nn.Sequential(DSConvBlock(4 * n, 2 * n), DSConvBlock(2 * n, 2 * n))
        self.up4 = nn.ConvTranspose2d(2 * n, n, 2, stride=2)
        self.dec4 = nn.Sequential(DSConvBlock(2 * n, n), DSConvBlock(n, n))

        self.head = nn.Conv2d(n, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d1 = self.dec1(torch.cat([self.up1(b), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))

        return torch.softmax(self.head(d4), dim=1)


class UNetDSWUnidirectional(nn.Module):
    """Unidirectional UNet with depth-separable convolutions.

    Corresponds to UNETdswUnidirectional in NeuralNetworkDefs.wl lines 130-161.
    """

    def __init__(self, n: int = 64, in_channels: int = 1):
        super().__init__()
        self.enc1 = nn.Sequential(
            BasicBlock(in_channels, n), DSConvBlock(n, n), nn.Dropout2d(0.1)
        )
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Sequential(
            DSConvBlock(n, 2 * n), DSConvBlock(2 * n, 2 * n), nn.Dropout2d(0.1)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Sequential(
            DSConvBlock(2 * n, 4 * n), DSConvBlock(4 * n, 4 * n), nn.Dropout2d(0.1)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        self.enc4 = nn.Sequential(
            DSConvBlock(4 * n, 8 * n), DSConvBlock(8 * n, 8 * n), nn.Dropout2d(0.1)
        )
        self.pool4 = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Sequential(
            DSConvBlock(8 * n, 16 * n), DSConvBlock(16 * n, 16 * n), nn.Dropout2d(0.2)
        )

        self.up1 = nn.ConvTranspose2d(16 * n, 8 * n, 2, stride=2)
        self.dec1 = nn.Sequential(DSConvBlock(16 * n, 8 * n), DSConvBlock(8 * n, 8 * n))
        self.up2 = nn.ConvTranspose2d(8 * n, 4 * n, 2, stride=2)
        self.dec2 = nn.Sequential(DSConvBlock(8 * n, 4 * n), DSConvBlock(4 * n, 4 * n))
        self.up3 = nn.ConvTranspose2d(4 * n, 2 * n, 2, stride=2)
        self.dec3 = nn.Sequential(DSConvBlock(4 * n, 2 * n), DSConvBlock(2 * n, 2 * n))
        self.up4 = nn.ConvTranspose2d(2 * n, n, 2, stride=2)
        self.dec4 = nn.Sequential(DSConvBlock(2 * n, n), DSConvBlock(n, n))

        self.head_ant = nn.Conv2d(n, 2, 1)
        self.head_ret = nn.Conv2d(n, 2, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d1 = self.dec1(torch.cat([self.up1(b), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d1), e3], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d2), e2], dim=1))
        d4 = self.dec4(torch.cat([self.up4(d3), e1], dim=1))

        return {
            "ant": torch.softmax(self.head_ant(d4), dim=1),
            "ret": torch.softmax(self.head_ret(d4), dim=1),
        }
