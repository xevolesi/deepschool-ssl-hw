import os

import addict
import pandas as pd
import pytest

from source.utils.general import read_config


@pytest.fixture(scope="session")
def get_test_config_path() -> str:
    return "config.yml"


@pytest.fixture(scope="module")
def get_test_config(get_test_config_path: str) -> addict.Dict:
    config = read_config(get_test_config_path)
    config.dataset.image_folder = "test_data/images"
    config.dataset.csv_path = "test_data/test.csv"

    # Low batch size and 0 workers for CI.
    config.training.batch_size = 4
    config.training.dataloader_num_workers = 0

    # Disable weight downloading for tests.
    config.model.timm_pretrained = False
    return config


@pytest.fixture(scope="module")
def get_test_csv(get_test_config) -> pd.DataFrame:
    return pd.read_csv(get_test_config.dataset.csv_path)


@pytest.fixture(scope="session")
def get_test_extended_settings() -> tuple[bool, bool]:
    return tuple(map(bool, (os.getenv("IS_LOCAL_RUN"), os.getenv("ENABLE_TESTS_ON_GPU"))))
