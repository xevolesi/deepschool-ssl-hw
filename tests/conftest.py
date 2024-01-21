import addict
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
    return config