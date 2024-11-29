from mlagents.torch_utils import torch, nn
from mlagents.trainers.torch_entities.layers import LinearEncoder, linear_layer
from typing import List
import math

class BrainMLP(nn.Module):
    MODEL_EXPORT_VERSION = 3  # Corresponds to ModelApiVersion.MLAgents2_0

    def __init__(
        self,
        input_size: int,
        aggregation_layers: int,
        hidden_size: int,
        feature_selection_layers: int,
        output_sizes: List[int]
    ):
        super().__init__()

        self.feature_aggregator = LinearEncoder(input_size, aggregation_layers, hidden_size)
        feature_selectors = []

        for shape in output_sizes:
            selector_layers = []
            
            for _ in range(feature_selection_layers - 1):
                selector_layers.append(linear_layer(hidden_size, hidden_size))
            selector_layers.append(linear_layer(hidden_size, shape))

            feature_selector = torch.nn.Sequential(*selector_layers)
            feature_selectors.append(feature_selector)
        self.feature_selectors = nn.ModuleList(feature_selectors)

        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False
        )

    def forward(self, input_tensor) -> List[torch.Tensor]:
        features = self.feature_aggregator(input_tensor)
        return [feature_selector(features) for feature_selector in self.feature_selectors]
    
class HardSelector(nn.Module):
    MODEL_EXPORT_VERSION = 3  # Corresponds to ModelApiVersion.MLAgents2_0

    def __init__(
        self,
        input_shape,
        selection_part: float
    ):
        super().__init__()

        self.feature_importance = torch.nn.Parameter(torch.ones(input_shape, dtype=torch.float))
        self.selection_part = int(selection_part * input_shape)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        features_weighted = input_tensor * self.feature_importance
        topk = torch.topk(features_weighted, self.selection_part, dim=1).values
        threshold = topk[:, -1].unsqueeze(1).expand(-1, input_tensor.shape[1])

        features_cropped = torch.where(features_weighted > threshold, input_tensor, torch.zeros_like(input_tensor))
        return features_cropped
    
class BrainHardSelection(nn.Module):
    MODEL_EXPORT_VERSION = 3  # Corresponds to ModelApiVersion.MLAgents2_0

    def __init__(
        self,
        input_size: int,
        aggregation_layers: int,
        hidden_size: int,
        feature_selection_layers: int,
        output_sizes: List[int]
    ):
        super().__init__()

        self.feature_aggregator = LinearEncoder(input_size, aggregation_layers, hidden_size)
        feature_selectors = []

        for shape in output_sizes:
            selector_layers = []
            
            selector_layers.append(HardSelector(input_size, 0.3))
            for _ in range(feature_selection_layers - 1):
                selector_layers.append(linear_layer(hidden_size, hidden_size))
            selector_layers.append(linear_layer(hidden_size, shape))

            feature_selector = torch.nn.Sequential(*selector_layers)
            feature_selectors.append(feature_selector)
        self.feature_selectors = nn.ModuleList(feature_selectors)

        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False
        )

    def forward(self, input_tensor) -> List[torch.Tensor]:
        features = self.feature_aggregator(input_tensor)
        return [feature_selector(features) for feature_selector in self.feature_selectors]