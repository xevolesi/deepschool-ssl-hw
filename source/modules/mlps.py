from torch import nn


class Projector(nn.Sequential):
    """Описание смотрите в статье на странице 3 в Projection MLP."""

    def __init__(self, in_features: int, out_features: int) -> None:
        self.in_features: int = in_features
        self.out_features: int = out_features
        super().__init__(
            nn.Linear(in_features, in_features, bias=False),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, in_features, bias=False),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, out_features, bias=False),
            nn.BatchNorm1d(out_features, affine=False),
        )


class Predictor(nn.Sequential):
    """Описание смотрите в статье на странице 3 в Prediction MLP."""

    def __init__(self, in_features: int, predictor_features: int, out_features: int) -> None:
        self.in_features: int = in_features
        self.out_features: int = out_features
        self.predictor_features: int = predictor_features
        super().__init__(
            nn.Linear(in_features, predictor_features, bias=False),
            nn.BatchNorm1d(predictor_features),
            nn.ReLU(inplace=True),
            nn.Linear(predictor_features, out_features),
        )
