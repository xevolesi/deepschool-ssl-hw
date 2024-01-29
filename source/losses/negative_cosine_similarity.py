import torch
from torch import nn
from torch.nn.functional import cosine_similarity


class NegativeCosineSimilarity(nn.Module):
    """
    Подробнее можно прочитать в статье на странице 2 в параграфе 3.
    Данный класс реализует уравнение (2) из статьи на странице 3.
    """

    def __init__(self, dim: int = 1, eps: float = 1e-8) -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps

    def forward(
        self,
        view1_embeddings: torch.Tensor,
        view2_embeddings: torch.Tensor,
        view1_predictions: torch.Tensor,
        view2_predictions: torch.Tensor,
    ) -> torch.Tensor:
        view1_loss = cosine_similarity(view1_predictions, view2_embeddings, self.dim, self.eps).mean()
        view2_loss = cosine_similarity(view2_predictions, view1_embeddings, self.dim, self.eps).mean()
        return -0.5 * (view1_loss + view2_loss)
