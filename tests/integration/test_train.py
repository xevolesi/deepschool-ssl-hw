import pathlib
import shutil
from copy import deepcopy
from datetime import datetime

import addict
import pandas as pd
import pytest
import pytz

from source.utils.training import train


def test_train(
    get_test_config: addict.Dict, get_test_csv: pd.DataFrame, get_test_extended_settings: tuple[bool, bool]
) -> None:
    config = deepcopy(get_test_config)
    is_local_run, enable_tests_on_gpu = get_test_extended_settings
    if is_local_run is None:
        pytest.skip("Skipping tests for training procedure in CI")
    if enable_tests_on_gpu:
        config.training.device = "cuda:0"
    run_id = datetime.now(tz=pytz.utc).strftime("%m-%d-%Y-%H-%M-%S")
    base_log_dir = pathlib.Path(config.logs.checkpoint_folder) / run_id
    base_log_dir.mkdir(exist_ok=True, parents=True)
    train(get_test_csv, config, None, base_log_dir)
    assert base_log_dir / f"best_checkpoint_{config.model.backbone_name}.pth"

    shutil.rmtree(base_log_dir.as_posix())
    assert not base_log_dir.exists()
