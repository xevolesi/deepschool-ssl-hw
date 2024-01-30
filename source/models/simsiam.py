import timm
import torch
from torch import nn

from source.modules.mlps import Predictor, Projector
from source.utils.general import get_cpu_state_dict


class TimmSimSiam(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnextv2_base",
        projection_dim: int = 2048,
        prediction_dim: int = 512,
        timm_pretrained: bool = False,
    ) -> None:
        super().__init__()

        # Здесь (https://huggingface.co/docs/timm/feature_extraction)
        # можно почитать подробнее про то, как пользоваться моделями
        # из `timm` в качестве экстракторов признаков.
        self.backbone: nn.Module = timm.create_model(
            backbone_name, pretrained=timm_pretrained, exportable=True, num_classes=0
        )

        # Это - размерность вектора-признака после глобального пулинга.
        # В лекции мы договорились называть этот вектор - вектором
        # репрезентаций.
        self.representation_dim = self.backbone.num_features

        # Это - размерность вектора-признака после проекции.
        # В лекции мы договорились называть этот вектор эмбеддингом.
        self.projection_dim = projection_dim
        self.projector = Projector(self.representation_dim, self.projection_dim)

        self.prediction_dim = prediction_dim
        self.predictor = Predictor(self.projection_dim, self.prediction_dim, self.projection_dim)

    def get_representations(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.backbone(tensor)

    def get_projections(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.projector(self.backbone(tensor))

    def get_predictions(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.predictor(self.projector(self.backbone(tensor)))

    def forward(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.get_projections(tensor)
        predictions = self.predictor(embeddings)
        return embeddings.detach(), predictions

    def get_backbone_state_dict(self) -> dict[str, torch.Tensor]:
        return get_cpu_state_dict(self.backbone)
