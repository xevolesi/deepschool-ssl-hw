import os
import pathlib

import pandas as pd
import torch

try:
    import wandb
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
    wandb = None

from source.utils.general import read_config
from source.utils.training import train

# Set benchmark to True and deterministic to False
# if you want to speed up training with less level of reproducibility.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Speed up GEMM if GPU allowed to use TF32.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Load envvars from .env.
if load_dotenv is not None and pathlib.Path(".env").exists():
    load_dotenv(".env")


def main():
    config = read_config("config.yml")
    if wandb is None or os.getenv("WANDB_API_KEY") is None:
        config.training.use_wandb = False
    run = wandb.init(project="SimSiam", config=config) if config.training.use_wandb else None
    dataframe = pd.read_csv(config.dataset.csv_path)
    train(dataframe, config, run)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    main()
