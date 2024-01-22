import addict
import albumentations as album
from albumentations.core.serialization import Serializable


def get_albumentation_augs(config: addict.Dict) -> dict[str, Serializable]:
    """Build albumentations's augmentation pipelines from configuration file."""
    transforms = {}
    for subset in config.augmentations:
        transforms[subset] = album.from_dict(config.augmentations[subset])

    return transforms
