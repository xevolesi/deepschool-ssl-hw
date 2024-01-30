import sys
import typing as ty

import addict
import pandas as pd
import torch
from loguru import logger
from torch.utils.data import DataLoader

try:
    from wandb.wandb_run import Run
except ImportError:
    Run = ty.Any

from source.datasets.ssl_dataset import build_dataloaders
from source.losses import NegativeCosineSimilarity
from source.models import TimmSimSiam
from source.utils.augmentations import get_albumentation_augs
from source.utils.evaluation import linear_probing
from source.utils.general import get_object_from_dict

logger.remove()
logger.add(
    sys.stdout,
    format=(
        "[<green>{time: HH:mm:ss}</green> | <blue>{level}</blue> | "
        "<magenta>training.py</magenta>:<yellow>{line}</yellow>] {message}"
    ),
    level="INFO",
    colorize=True,
)


def train(dataframe: pd.DataFrame, config: addict.Dict, wb_run: Run | None) -> None:
    augs = get_albumentation_augs(config)
    device = torch.device(config.training.device)
    dataloaders = build_dataloaders(dataframe, config)
    criterion: NegativeCosineSimilarity = get_object_from_dict(config.criterion)
    model: TimmSimSiam = get_object_from_dict(config.model).to(device)
    optimizer: torch.optim.Optimizer = get_object_from_dict(config.optimizer, params=model.parameters())
    scheduler: torch.optim.lr_scheduler.LRScheduler = get_object_from_dict(
        config.scheduler, optimizer=optimizer, T_max=config.training.epochs
    )

    best_knn_acc = 0.0
    best_weights = None
    for epoch in range(config.training.epochs):
        dataloaders["train"].dataset.switch_transforms(augs["train"])
        training_loss = train_one_epoch(model, dataloaders["train"], optimizer, criterion, device)

        if scheduler is not None:
            scheduler.step()

        # Мы хотим проводить linear probing на каждой эпохе с помощью
        # нашей обученной модели. Linear probing заключается в том, что
        # мы будем брать KNN, фитить его на репрезентациях, полученных
        # из тренировочной выборки, а предсказывать будем на
        # репрезентациях, полученных из валидационной выборки. Поэтому
        # для тренировочного даталоадера нам нужно убрать все
        # аугментации, кроме перевода в тензор и нормализации, т.к. мы
        # в момент linear probing'a мы хотим работать с репрезентациями,
        # полученными из чистых изображений. Метрикой качества нашей
        # процедуры linear probing будет обычная топ-1 accuracy.
        # Выбор accuracy обусловлен лишь ее простотой.
        dataloaders["train"].dataset.switch_transforms(augs["val"])
        val_accuracy = linear_probing(model, dataloaders["train"], dataloaders["val"], device)

        logger.info(
            "[EPOCH {epoch}/{total_epochs}] TL={tl:.2f}, KNN Acc@1={knn_acc:.3f}",
            epoch=epoch + 1,
            total_epochs=config.training.epochs,
            tl=training_loss,
            knn_acc=val_accuracy,
        )

        if wb_run is not None:
            wb_run.log(
                {
                    "training_loss": training_loss,
                    "knn_acc": val_accuracy,
                    "epoch": epoch,
                    "LR": scheduler.get_last_lr()[0],
                }
            )
        if val_accuracy >= best_knn_acc:
            best_knn_acc = val_accuracy
            best_weights = model.get_backbone_state_dict()
    logger.info("Best result with KNN Acc@1={knn_acc:.3f} on validation set", knn_acc=best_knn_acc)

    # Прогоним linear probing на тестовых данных.
    model.backbone.load_state_dict(best_weights)
    test_accuracy = linear_probing(model, dataloaders["train"], dataloaders["test"], device)
    logger.info("Best weights provides KNN Acc@1={knn_acc:.3f} on test set", knn_acc=test_accuracy)


def train_one_epoch(
    model: TimmSimSiam,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: NegativeCosineSimilarity,
    device: torch.device,
) -> float:
    model.train()
    training_loss = torch.as_tensor(0.0, device=device)
    for batch in loader:
        view1 = batch["view1"].to(device, non_blocking=True)
        view2 = batch["view2"].to(device, non_blocking=True)
        view1_embeddings, view1_predictions = model(view1)
        view2_embeddings, view2_predictions = model(view2)
        loss = criterion(view1_embeddings, view2_embeddings, view1_predictions, view2_predictions)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        training_loss += loss
    return training_loss.detach().cpu().item() / len(loader)
