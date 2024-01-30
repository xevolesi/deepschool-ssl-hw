import torch
from torch.utils.data import DataLoader

from source.models import EuclideanKNN, TimmSimSiam


@torch.no_grad()
def gather_representations_and_labels(
    model: TimmSimSiam, loader: DataLoader, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    representations_bank = torch.zeros((len(loader.dataset), model.representation_dim))
    labels_bank = torch.zeros((len(loader.dataset),))
    for batch_idx, batch in enumerate(loader):
        batch_slice = slice(batch_idx * loader.batch_size, (batch_idx + 1) * loader.batch_size)
        representations = model.get_representations(batch["view1"].to(device, non_blocking=True))
        representations_bank[batch_slice] = representations.detach().cpu()
        labels_bank[batch_slice] = batch["label"]
    representations_bank = torch.nn.functional.normalize(representations_bank)
    return representations_bank, labels_bank


def linear_probing(model: TimmSimSiam, train_loader: DataLoader, val_loader: DataLoader, device: torch.device) -> float:
    val_representations, val_labels = gather_representations_and_labels(model, val_loader, device)
    train_representations, train_labels = gather_representations_and_labels(model, train_loader, device)
    knn = EuclideanKNN()
    knn.fit(train_representations, train_labels)
    predictions = knn.predict(val_representations)
    return (val_labels == predictions).float().mean().item()
