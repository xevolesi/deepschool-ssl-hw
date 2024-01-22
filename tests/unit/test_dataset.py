import pathlib
from itertools import product

import addict
import numpy as np
import pandas as pd
import pytest
import torch

from source.datasets.ssl_dataset import SSLDataset
from source.datasets.utils import read_image
from source.utils.augmentations import get_albumentation_augs
from source.utils.custom_types import DataPoint


def test_read_image(get_test_config: addict.Dict) -> None:
    for image_path in pathlib.Path(get_test_config.dataset.image_folder).iterdir():
        image = read_image(image_path)
        assert image is not None


def __assert_numpy_datapoint(data_point: DataPoint) -> None:
    assert data_point["view1"] is None
    assert data_point["view2"] is None
    assert isinstance(data_point["image"], np.ndarray)
    assert data_point["image"].dtype == np.uint8
    assert isinstance(data_point["label"], int)


def __assert_torch_datapoint(data_point: DataPoint) -> None:
    for view_name in ("view1", "view2"):
        assert isinstance(data_point[view_name], torch.Tensor)
        assert data_point[view_name].dtype == torch.float32
        assert data_point[view_name].shape == (3, 224, 224)
    assert isinstance(data_point["image"], np.ndarray)
    assert data_point["image"].dtype == np.uint8
    assert isinstance(data_point["label"], int)


@pytest.mark.parametrize(("use_cache", "use_transforms"), product((True, False), (True, False)))
def test_dataset(
    use_cache: bool, use_transforms: bool, get_test_config: addict.Dict, get_test_csv: pd.DataFrame
) -> None:
    get_test_config.training.use_image_caching = use_cache
    transforms = None
    if use_transforms:
        transforms = get_albumentation_augs(get_test_config)["train"]
    dataset = SSLDataset(get_test_config, get_test_csv, transforms)
    assert len(dataset) == len(get_test_csv)

    # Quite bad practice to access protected method from outside
    # but ... :)
    for image in dataset._SSLDataset__image_collection:
        if use_cache:
            assert isinstance(image, np.ndarray)
        else:
            assert isinstance(image, str)

    for item in dataset:
        if not use_transforms:
            __assert_numpy_datapoint(item)
        else:
            __assert_torch_datapoint(item)
