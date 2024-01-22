import pathlib

import addict
import pandas as pd
from torch.utils.data import Dataset

from source.utils.custom_types import DataPoint

from .utils import read_image


class SSLDataset(Dataset):
    def __init__(self, config: addict.Dict, dataframe: pd.DataFrame, transforms=None) -> None:
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
            data_point["view2"] = self.transforms(image=image)["image"]
        return data_point
