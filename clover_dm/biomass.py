"""Direct multi-target biomass composition regressor."""

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2


class BiomassRegressor(nn.Module):
    """Predict a physically consistent five-value biomass composition.

    The network predicts the four disjoint components grass, white clover,
    red clover and weeds. A softmax makes them sum to one, and total clover is
    derived as white + red. The output order is grass, clover, white clover,
    red clover and weeds.
    """

    def __init__(
        self,
        pretrained_backbone: bool = True,
        dropout: float = 0.3,
        hidden_channels: int = 512,
    ) -> None:
        super().__init__()
        weights: Optional[MobileNet_V2_Weights]
        weights = MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        backbone = mobilenet_v2(weights=weights)
        self.features = backbone.features
        feature_channels = backbone.last_channel
        self.head = nn.Sequential(
            nn.Linear(feature_channels * 2, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 4),
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        average = F.adaptive_avg_pool2d(features, 1).flatten(1)
        maximum = F.adaptive_max_pool2d(features, 1).flatten(1)
        components = self.head(torch.cat((average, maximum), dim=1)).softmax(dim=1)
        grass, white, red, weeds = components.unbind(dim=1)
        return torch.stack((grass, white + red, white, red, weeds), dim=1)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
