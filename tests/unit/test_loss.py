import pytest
import torch

from source.losses import NegativeCosineSimilarity

EPS: float = 1e-6

__TEST_CASES = [
    ([[0, 1]], [[1, 0]], 0),
    ([[1, 1]], [[1, 1]], -1.0),
    ([[0, 1]], [[0, -1]], 1.0),
]


@pytest.mark.parametrize(("vec1", "vec2", "expected"), __TEST_CASES)
def test_neg_cos_sim(vec1, vec2, expected) -> None:
    criterion = NegativeCosineSimilarity()
    vec1 = torch.as_tensor(vec1, dtype=torch.float32, device=torch.device("cpu"))
    vec2 = torch.as_tensor(vec2, dtype=torch.float32, device=torch.device("cpu"))
    assert abs(criterion(vec1, vec1, vec2, vec2).item() - expected) < EPS
