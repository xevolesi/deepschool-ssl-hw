import pathlib
from functools import partial

import addict
import pandas as pd
import torch
from albumentations import Compose
from albumentations.core.serialization import Serializable
from torch.utils.data import DataLoader, Dataset

from source.utils.augmentations import get_albumentation_augs
from source.utils.custom_types import DataPoint

from .utils import fix_worker_seeds, read_image, ssl_collate_fn, train_val_test_split


class SSLDataset(Dataset):
    def __init__(
        self,
        config: addict.Dict,
        dataframe: pd.DataFrame,
        subset: str = "train",
        transforms: Compose | Serializable | None = None,
    ) -> None:
        self.__subset = subset
        self.__image_dir = config.dataset.image_folder
        self.__label_mapper = config.dataset.label_mapping
        self.__labels = dataframe["label"].map(self.__label_mapper).to_numpy()
        self.__image_collection = (
            dataframe["image"]
            .apply(lambda image_hash: pathlib.Path(self.__image_dir).joinpath(image_hash).as_posix() + ".jpg")
            .to_numpy()
        )
        self.transforms = transforms
        # Let's use poorman image caching.
        self.__cached = config.training.use_image_caching
        if self.__cached:
            self.__image_collection = list(map(read_image, self.__image_collection))

    def __len__(self) -> int:
        return len(self.__image_collection)

    def __getitem__(self, index: int) -> DataPoint:
        if self.__cached:
            image = self.__image_collection[index]
        else:
            image_path = self.__image_collection[index]
            image = read_image(image_path)
        label = self.__labels[index].item()

        data_point = {"view1": None, "view2": None, "image": image, "label": label}
        if self.transforms is not None:
            data_point["view1"] = self.transforms(image=image)["image"]
            if self.__subset == "train":
                data_point["view2"] = self.transforms(image=image)["image"]
        return data_point


def build_dataloaders(dataframe: pd.DataFrame, config: addict.Dict) -> dict[str, DataLoader]:
    split = train_val_test_split(dataframe, config)
    transforms = get_albumentation_augs(config)
    dataloaders = {}
    for subset_name in split:
        dataloaders[subset_name] = DataLoader(
            SSLDataset(config, split[subset_name], subset_name, transforms[subset_name]),
            config.training.batch_size,
            shuffle=subset_name == "train",
            pin_memory=torch.cuda.is_available(),
            num_workers=config.training.dataloader_num_workers,
            collate_fn=partial(ssl_collate_fn, subset=subset_name),
            worker_init_fn=fix_worker_seeds,
        )
    return dataloaders
