import torch
import torch.nn as nn


def _activation_block(hidden_size: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(0.1),
    )


class LSTMRegressor(nn.Module):
    """Sequence-to-one LSTM regressor."""

    def __init__(self, input_size: int, hidden_size: int = 96, num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            _activation_block(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, features]
        output, _ = self.lstm(x)
        last_timestep = output[:, -1, :]
        return self.head(last_timestep)


class GRURegressor(nn.Module):
    """GRU alternative for comparison."""

    def __init__(self, input_size: int, hidden_size: int = 96, num_layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.head = nn.Sequential(
            _activation_block(hidden_size),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.gru(x)
        last_timestep = output[:, -1, :]
        return self.head(last_timestep)


class TemporalCNNRegressor(nn.Module):
    """Dilated 1D CNN for temporal patterns."""

    def __init__(self, input_size: int, channels: list[int] | None = None, kernel_size: int = 3, dropout: float = 0.25):
        super().__init__()
        channels = channels or [64, 128]

        layers = []
        in_channels = input_size
        dilation = 1
        for out_channels in channels:
            padding = (kernel_size - 1) * dilation
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels
            dilation *= 2

        self.temporal_blocks = nn.Sequential(*layers)
        self.head = nn.Sequential(
            _activation_block(in_channels),
            nn.Linear(in_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, features] -> [batch, features, seq]
        x = x.permute(0, 2, 1)
        conv_out = self.temporal_blocks(x)
        last_step = conv_out[:, :, -1]
        return self.head(last_step)
