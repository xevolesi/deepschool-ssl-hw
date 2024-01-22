import pydoc

import addict
import albumentations as album
from albumentations.core.serialization import Serializable

from source.utils.augmentations import get_albumentation_augs


def _assert_augs_from_config(config: addict.Dict, subset_name: str, actual_augs: album.Compose) -> None:
    config_augs = config.augmentations[subset_name]
    assert config_augs["transform"]["__class_fullname__"] == "Compose"
    config_augs = config_augs["transform"]["transforms"]
    assert len(config_augs) == len(actual_augs)
    for config_aug, actual_aug in zip(config_augs, actual_augs, strict=True):
        # Hack with type(...) | Serializable specially for MyPy. :)
        assert isinstance(actual_aug, type(pydoc.locate(config_aug["__class_fullname__"])) | Serializable)


def test_get_albumentation_augs(get_test_config: addict.Dict) -> None:
    augs = get_albumentation_augs(get_test_config)
    assert set(augs.keys()) == set(get_test_config.augmentations.keys())
    for subset_name in augs:
        _assert_augs_from_config(get_test_config, subset_name, augs[subset_name])
