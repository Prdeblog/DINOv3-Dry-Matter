"""Improved dual-pooling channel-spatial attention."""

import torch
from torch import Tensor, nn


class DualPoolingAttention(nn.Module):
    """Fuse average/max descriptors and retain a scaled residual connection.

    The two pooling paths share their channel MLP to reduce parameters. A
    lightweight spatial gate is added after channel attention, and layer-scale
    initialization keeps the block close to identity at the start of transfer
    learning.
    """

    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        layer_scale_init: float = 1e-2,
    ) -> None:
        super().__init__()
        hidden_channels = max(in_channels // reduction_ratio, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_channel_mlp = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False),
        )
        self.spatial_gate = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.layer_scale = nn.Parameter(
            torch.full((1, in_channels, 1, 1), float(layer_scale_init))
        )

    def forward(self, input_feature: Tensor) -> Tensor:
        channel_attention = torch.sigmoid(
            self.shared_channel_mlp(self.avg_pool(input_feature))
            + self.shared_channel_mlp(self.max_pool(input_feature))
        )
        channel_refined = input_feature * channel_attention
        spatial_descriptor = torch.cat(
            (
                channel_refined.mean(dim=1, keepdim=True),
                channel_refined.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_attention = torch.sigmoid(self.spatial_gate(spatial_descriptor))
        return input_feature + self.layer_scale * channel_refined * spatial_attention


if __name__ == "__main__":
    block = DualPoolingAttention(32)
    output = block(torch.randn(2, 32, 50, 50))
    print(output.shape)
