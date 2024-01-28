import timm
import torch
from torch import nn

from source.modules.mlps import Predictor, Projector
from source.utils.custom_types import SimSiamOutput


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
        self.backbone = timm.create_model(backbone_name, pretrained=timm_pretrained, exportable=True, num_classes=0)

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

    def forward(self, view1: torch.Tensor, view2: torch.Tensor) -> SimSiamOutput:
        embedding1 = self.projector(self.backbone(view1))
        embedding2 = self.projector(self.backbone(view2))
        prediction1 = self.predictor(embedding1)
        prediction2 = self.predictor(embedding2)
        return {
            "view1_predictions": prediction1,
            "view2_predictions": prediction2,
            "view1_embeddings": embedding1.detach(),
            "view2_embeddings": embedding2.detach(),
        }
