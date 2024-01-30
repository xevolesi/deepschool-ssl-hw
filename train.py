import pandas as pd
import torch

from source.utils.general import read_config
from source.utils.training import train

# Set benchmark to True and deterministic to False
# if you want to speed up training with less level of reproducibility.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Speed up GEMM if GPU allowed to use TF32.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    config = read_config("config.yml")
    dataframe = pd.read_csv(config.dataset.csv_path)
    train(dataframe, config)


if __name__ == "__main__":
    main()
