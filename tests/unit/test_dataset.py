import pathlib

import addict

from source.datasets.utils import read_image


def test_read_image(get_test_config: addict.Dict) -> None:
    for image_path in pathlib.Path(get_test_config.dataset.image_folder).iterdir():
        image = read_image(image_path)
        assert image is not None
