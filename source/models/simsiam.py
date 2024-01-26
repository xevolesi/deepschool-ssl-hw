import timm
import torch
from torch import nn

from source.modules.mlps import Predictor, Projector
from source.utils.custom_types import SimSiamOutput


class TimmSimSiam(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnextv2_base",
        feature_dim: int = 2048,
        predictor_dim: int = 512,
        timm_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone_name, timm_pretrained, exportable=True)
        self.embedding_dim = self.backbone.num_features
        self.feature_dim = feature_dim
        self.predictor_dim = predictor_dim

        # Throw away classification head since we need only feature
        # extractor.
        self.backbone.reset_classifier(num_classes=0)

        self.projector = Projector(self.embedding_dim, self.feature_dim)
        self.predictor = Predictor(self.embedding_dim, self.predictor_dim, self.feature_dim)

    def get_representations(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.backbone(tensor)

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
