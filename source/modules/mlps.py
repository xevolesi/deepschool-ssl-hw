from torch import nn


class Projector(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        super().__init__(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, out_channels, bias=False),
            nn.BatchNorm1d(out_channels, affine=False),
        )


class Predictor(nn.Sequential):
    def __init__(self, in_channels: int, prediction_channels: int, out_channels: int) -> None:
        self.in_channels: int = in_channels
        self.prediction_channels: int = prediction_channels
        self.out_channels: int = out_channels
        super().__init__(
            nn.Linear(in_channels, prediction_channels, bias=False),
            nn.BatchNorm1d(prediction_channels),
            nn.ReLU(inplace=True),
            nn.Linear(prediction_channels, out_channels),
        )
