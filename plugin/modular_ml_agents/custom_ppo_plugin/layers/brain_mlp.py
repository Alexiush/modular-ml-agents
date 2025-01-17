from mlagents.torch_utils import torch, nn
from mlagents.trainers.torch_entities.layers import LinearEncoder, linear_layer, LSTM
from typing import List

class BrainMLP(nn.Module):
    MODEL_EXPORT_VERSION = 3  # Corresponds to ModelApiVersion.MLAgents2_0

    def __init__(
        self,
        input_size: int,
        aggregation_layers: int,
        hidden_size: int,
        feature_selection_layers: int,
        output_sizes: List[int],
        output_channels: List[int]
    ):
        super().__init__()

        self.feature_aggregator = LinearEncoder(input_size, aggregation_layers, hidden_size)
        feature_selectors = []

        for shape, ch in zip(output_sizes, output_channels):
            selector_layers = []
            
            for _ in range(feature_selection_layers - 1):
                selector_layers.append(linear_layer(hidden_size, hidden_size))
            selector_layers.append(linear_layer(hidden_size, shape * ch))

            feature_selector = torch.nn.Sequential(*selector_layers)
            feature_selectors.append(feature_selector)
        self.feature_selectors = nn.ModuleList(feature_selectors)

        self.output_channels = output_channels

        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False
        )

    def forward(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        features = self.feature_aggregator(input_tensor)
        return [feature_selector(features).view(features.shape[0], self.output_channels[i], -1) for i, feature_selector in enumerate(self.feature_selectors)]
    
class HardSelector(nn.Module):
    MODEL_EXPORT_VERSION = 3  # Corresponds to ModelApiVersion.MLAgents2_0

    def __init__(
        self,
        input_shape: int,
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
        output_sizes: List[int],
        output_channels: List[int]
    ):
        super().__init__()

        self.feature_aggregator = LinearEncoder(input_size, aggregation_layers, hidden_size)
        feature_selectors = []

        for shape, ch in zip(output_sizes, output_channels):
            selector_layers = []
            
            selector_layers.append(HardSelector(hidden_size, 0.3))
            for _ in range(feature_selection_layers - 1):
                selector_layers.append(linear_layer(hidden_size, hidden_size))
            selector_layers.append(linear_layer(hidden_size, shape * ch))

            feature_selector = torch.nn.Sequential(*selector_layers)
            feature_selectors.append(feature_selector)
        self.feature_selectors = nn.ModuleList(feature_selectors)

        self.output_channels = output_channels

        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False
        )

    def forward(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        features = self.feature_aggregator(input_tensor)
        return [feature_selector(features).view(features.shape[0], self.output_channels[i], -1) for i, feature_selector in enumerate(self.feature_selectors)]
    
class BrainRNN(nn.Module):
    MODEL_EXPORT_VERSION = 3  # Corresponds to ModelApiVersion.MLAgents2_0

    def __init__(
        self,
        input_size: int,
        aggregation_layers: int,
        hidden_size: int,
        memory_size: int,
        feature_selection_layers: int,
        output_sizes: List[int],
        output_channels: List[int]
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.rnn_hidden_size = memory_size // 2

        self.feature_aggregator = LinearEncoder(input_size, aggregation_layers, self.rnn_hidden_size)
        feature_selectors = []

        for shape, ch in zip(output_sizes, output_channels):
            selector_layers = []
            
            selector_layers.append(linear_layer(self.rnn_hidden_size, hidden_size))
            for _ in range(feature_selection_layers - 2):
                selector_layers.append(linear_layer(hidden_size, hidden_size))
            selector_layers.append(linear_layer(hidden_size, shape * ch))

            feature_selector = torch.nn.Sequential(*selector_layers)
            feature_selectors.append(feature_selector)
        self.feature_selectors = nn.ModuleList(feature_selectors)

        self.lstm = LSTM(self.rnn_hidden_size, memory_size)
        self.memories = torch.zeros([1, memory_size])

        self.output_channels = output_channels

        self.version_number = torch.nn.Parameter(
            torch.Tensor([self.MODEL_EXPORT_VERSION]), requires_grad=False
        )

    def forward(self, input_tensor: torch.Tensor, memories_mask: torch.Tensor) -> List[torch.Tensor]:
        features = self.feature_aggregator(input_tensor)

        if len(memories_mask.shape) > 0:
            mask = torch.ones([self.memories.shape[0], features.shape[0], self.memories.shape[1]], dtype=torch.int32).detach()
            masked_columns = memories_mask[0, 0].view(1).int()

            mask[:, :, :masked_columns] = 0
            memories_masked = self.memories * mask

            features_out, memories = self.lstm(features.contiguous().view([-1, 1, self.rnn_hidden_size]), memories_masked)
            self.memories = memories[:, -1, :]
        else:
            features_out = features.contiguous()

        features_out = features.contiguous()

        return [feature_selector(features_out).view(features.shape[0], self.output_channels[i], -1) for i, feature_selector in enumerate(self.feature_selectors)]
