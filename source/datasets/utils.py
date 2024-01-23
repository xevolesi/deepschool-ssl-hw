import random

import addict
import cv2
import jpeg4py as jpeg
import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from torch.utils.data import default_collate

from source.utils.custom_types import DataPoint, ImageDataPoint


def read_image(image_path: str) -> NDArray[np.uint8]:
    try:
        image = jpeg.JPEG(image_path).decode()
    except jpeg.JPEGRuntimeError:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def train_val_test_split(dataframe: pd.DataFrame, config: addict.Dict) -> dict[str, pd.DataFrame]:
    return {
        "train": dataframe.query(f"fold in {config.training.folds.train}").reset_index(drop=True),
        "val": dataframe.query(f"fold in {config.training.folds.val}").reset_index(drop=True),
        "test": dataframe.query(f"fold in {config.training.folds.test}").reset_index(drop=True),
    }


def ssl_collate_fn(batch: list[DataPoint]) -> dict[str, ImageDataPoint | torch.Tensor]:
    # We don't want to apply default collate to source images.
    images = [item["image"] for item in batch]
    collated_data = default_collate([item | {"image": 0} for item in batch])
    return collated_data | {"image": images}


def fix_worker_seeds(worker_id: int) -> None:
    """Fix seeds inside single worker."""
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.SystemRandom().seed(int(seed))
    torch.manual_seed(seed)
