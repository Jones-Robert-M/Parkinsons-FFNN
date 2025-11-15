"""Feedforward Neural Network for regression tasks."""

import torch
import torch.nn as nn
from typing import List, Optional


class FeedForwardNN(nn.Module):
    """
    Configurable feedforward neural network for regression.

    Args:
        input_dim: Number of input features
        hidden_dims: List of hidden layer dimensions
        output_dim: Number of output targets
        dropout_rate: Dropout probability (0 to disable)
        activation: Activation function ('relu', 'tanh', 'elu')
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        dropout_rate: float = 0.2,
        activation: str = "relu"
    ):
        super(FeedForwardNN, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # Choose activation function
        activation_map = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU(),
        }
        if activation not in activation_map:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation_fn = activation_map[activation]

        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            # Linear layer
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            # Batch normalization
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            # Activation
            layers.append(activation_map[activation])
            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # Output layer (no activation, no dropout)
        layers.append(nn.Linear(dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights using Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        return self.network(x)

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __str__(self) -> str:
        """String representation of the model."""
        arch = f"FeedForwardNN(\n"
        arch += f"  input_dim={self.input_dim},\n"
        arch += f"  hidden_dims={self.hidden_dims},\n"
        arch += f"  output_dim={self.output_dim},\n"
        arch += f"  total_params={self.get_num_parameters():,}\n"
        arch += ")"
        return arch
