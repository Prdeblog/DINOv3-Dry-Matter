"""Improved multi-scale hybrid cross block.

Compared with the original snippet, this implementation:
- supports different input/output channel counts;
- uses depthwise-separable 3x3/5x5 convolutions to control computation;
- uses GroupNorm, which is stable for small biomass-training batches;
- uses learnable residual scaling for safer insertion into a pretrained model.
"""

import math

import torch
from torch import Tensor, nn


def _group_count(channels: int, maximum: int = 8) -> int:
    return max(1, math.gcd(channels, maximum))


class DepthwiseConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class MultiScaleHybridCrossBlock(nn.Module):
    """Two-stage 3x3/5x5 cross fusion with a scaled residual path."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        layer_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.conv3_stage1 = DepthwiseConvNormAct(in_channels, out_channels, 3)
        self.conv5_stage1 = DepthwiseConvNormAct(in_channels, out_channels, 5)

        fused_channels = out_channels * 3
        self.fusion3 = nn.Sequential(
            nn.Conv2d(fused_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.fusion5 = nn.Sequential(
            nn.Conv2d(fused_channels, out_channels, 1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.conv3_stage2 = DepthwiseConvNormAct(out_channels, out_channels, 3)
        self.conv5_stage2 = DepthwiseConvNormAct(out_channels, out_channels, 5)

        self.bottleneck_fusion = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
        )
        self.layer_scale = nn.Parameter(
            torch.full((1, out_channels, 1, 1), float(layer_scale_init))
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.skip(x)
        feat3_stage1 = identity + self.conv3_stage1(x)
        feat5_stage1 = identity + self.conv5_stage1(x)
        stage1 = torch.cat((identity, feat3_stage1, feat5_stage1), dim=1)

        feat3_stage2 = feat3_stage1 + self.conv3_stage2(self.fusion3(stage1))
        feat5_stage2 = feat5_stage1 + self.conv5_stage2(self.fusion5(stage1))
        fused = self.bottleneck_fusion(
            torch.cat(
                (identity, feat3_stage1, feat5_stage1, feat3_stage2, feat5_stage2),
                dim=1,
            )
        )
        return identity + self.layer_scale * fused


if __name__ == "__main__":
    block = MultiScaleHybridCrossBlock(24, 32)
    output = block(torch.randn(2, 24, 64, 64))
    print(output.shape)
