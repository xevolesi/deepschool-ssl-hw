import pathlib

import addict
import yaml


def read_config(config_path: str) -> addict.Dict:
    with pathlib.Path(config_path).open("r") as cf:
        return addict.Dict(yaml.safe_load(cf))
