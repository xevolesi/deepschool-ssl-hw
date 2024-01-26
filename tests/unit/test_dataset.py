import pathlib
from itertools import product

import addict
import numpy as np
import pandas as pd
import pytest
import torch

from source.datasets.ssl_dataset import SSLDataset, build_dataloaders
from source.datasets.utils import read_image, train_val_test_split
from source.utils.augmentations import get_albumentation_augs
from source.utils.custom_types import DataPoint


def test_read_image(get_test_config: addict.Dict) -> None:
    for image_path in pathlib.Path(get_test_config.dataset.image_folder).iterdir():
        image = read_image(image_path)
        assert image is not None


def test_train_val_test_split(get_test_config: addict.Dict, get_test_csv: pd.DataFrame) -> None:
    split = train_val_test_split(get_test_csv, get_test_config)
    for subset_name in split:
        folds_from_config = getattr(get_test_config.training.folds, subset_name)
        folds_from_split = split[subset_name]["fold"].unique()
        assert set(folds_from_config) == set(folds_from_split)


def __assert_numpy_datapoint(data_point: DataPoint) -> None:
    assert data_point["view1"] is None
    assert data_point["view2"] is None
    assert isinstance(data_point["image"], np.ndarray)
    assert data_point["image"].dtype == np.uint8
    assert isinstance(data_point["label"], int)


def __assert_torch_datapoint(data_point: DataPoint, subset: str) -> None:
    def assert_torch_tensor(torch_tensor: torch.Tensor) -> None:
        assert isinstance(torch_tensor, torch.Tensor)
        assert torch_tensor.dtype == torch.float32
        assert torch_tensor.shape == (3, 224, 224)

    assert_torch_tensor(data_point["view1"])
    if subset == "train":
        assert_torch_tensor(data_point["view2"])
    else:
        assert data_point["view2"] is None
    assert isinstance(data_point["image"], np.ndarray)
    assert data_point["image"].dtype == np.uint8
    assert isinstance(data_point["label"], int)


@pytest.mark.parametrize(
    ("use_cache", "use_transforms", "subset"), product((True, False), (True, False), ("train", "val", "test"))
)
def test_dataset(
    use_cache: bool, use_transforms: bool, subset: str, get_test_config: addict.Dict, get_test_csv: pd.DataFrame
) -> None:
    get_test_config.training.use_image_caching = use_cache
    transforms = None
    if use_transforms:
        transforms = get_albumentation_augs(get_test_config)[subset]
    dataset = SSLDataset(get_test_config, get_test_csv, subset, transforms)
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
            __assert_torch_datapoint(item, subset)


def __assert_view(view: torch.Tensor, batch_size) -> None:
    assert view.shape == (batch_size, 3, 224, 224)
    assert isinstance(view, torch.Tensor)
    assert view.dtype == torch.float32


@pytest.mark.parametrize("use_cache", [True, False])
def test_build_dataloaders(use_cache: bool, get_test_config: addict.Dict, get_test_csv: pd.DataFrame) -> None:
    get_test_config.training.use_image_caching = use_cache
    dataloaders = build_dataloaders(get_test_csv, get_test_config)

    for subset_name, subset_loader in dataloaders.items():
        for batch in subset_loader:
            images = batch["image"]
            assert len(images) == get_test_config.training.batch_size
            assert all(isinstance(image, np.ndarray) for image in images)
            assert all(image.dtype == np.uint8 for image in images)
            __assert_view(batch["view1"], get_test_config.training.batch_size)
            if subset_name == "train":
                __assert_view(batch["view2"], get_test_config.training.batch_size)
            else:
                assert batch["view2"] is None
