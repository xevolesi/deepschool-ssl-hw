import torch


class EuclideanKNN:
    def __init__(self, k: int = 5) -> None:
        self.k = k
        self.train_features: torch.FloatTensor = None
        self.train_labels: torch.LongTensor = None

    def fit(self, features: torch.FloatTensor, labels: torch.LongTensor) -> None:
        self.train_labels = labels
        self.train_features = features

    @torch.inference_mode()
    def predict(self, features: torch.FloatTensor) -> torch.LongTensor:
        distances = torch.linalg.norm(features[:, None, :] - self.train_features[None, :, :], dim=-1)
        _, closest_samples_indices = torch.topk(distances, self.k, largest=False)
        labels, _ = self.train_labels[closest_samples_indices].mode()
        return labels
