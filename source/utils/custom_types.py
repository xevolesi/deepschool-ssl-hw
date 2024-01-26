import typing as ty

import numpy as np
import torch
from numpy.typing import NDArray

ImageDataPoint: ty.TypeAlias = NDArray[np.uint8] | torch.Tensor


class DataPoint(ty.TypedDict):
    view1: ImageDataPoint
    view2: ImageDataPoint
    image: ImageDataPoint
    label: int


class SimSiamOutput(ty.TypedDict):
    view1_embeddings: torch.FloatTensor
    view2_embeddings: torch.FloatTensor
    view1_predictions: torch.FloatTensor
    view2_predictions: torch.FloatTensor
