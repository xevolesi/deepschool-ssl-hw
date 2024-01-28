from itertools import product

import pytest
import torch

from source.modules.mlps import Predictor, Projector


def __assert_out_tensor(output: torch.Tensor, expected_out_shape: int) -> None:
    assert output.shape == expected_out_shape
    assert output.dtype == torch.float32


def __assert_mlp(mlp_cls: torch.nn.Module, mlp_kwargs, test_settings: tuple[bool, bool]) -> None:
    is_local_run, enable_tests_on_gpu = test_settings
    if not is_local_run:
        pytest.skip("Skipping modules test for CI")
    mlp = mlp_cls(**mlp_kwargs)
    random_tensor = torch.randn((2, mlp_kwargs["in_features"]), dtype=torch.float32).cpu()
    mlp.eval()
    with torch.no_grad():
        output = mlp(random_tensor)
        __assert_out_tensor(output, (2, mlp_kwargs["out_features"]))

    if enable_tests_on_gpu:
        mlp.train()
        mlp = mlp.cuda()
        random_tensor = random_tensor.cuda()
        with torch.no_grad():
            output = mlp(random_tensor)
            __assert_out_tensor(output, (2, mlp_kwargs["out_features"]))

        random_tensor.requires_grad = True
        output = mlp(random_tensor)
        __assert_out_tensor(output, (2, mlp_kwargs["out_features"]))

        # Assert gradients.
        dummy_loss = (1 - output).mean()
        dummy_loss.backward()
        assert random_tensor.grad is not None
        for param in mlp.parameters():
            assert param.grad is not None


@pytest.mark.parametrize(("in_features", "out_features"), product([128, 256, 512], [128, 256, 512]))
def test_projector(in_features: int, out_features: int, get_test_extended_settings: tuple[bool, bool]) -> None:
    __assert_mlp(Projector, {"in_features": in_features, "out_features": out_features}, get_test_extended_settings)


@pytest.mark.parametrize(
    ("in_features", "predictor_features", "out_features"), product([128, 256, 512], [128, 256, 512], [128, 256, 512])
)
def test_predictor(
    in_features: int, predictor_features: int, out_features: int, get_test_extended_settings: tuple[bool, bool]
) -> None:
    __assert_mlp(
        Predictor,
        {"in_features": in_features, "predictor_features": predictor_features, "out_features": out_features},
        get_test_extended_settings,
    )
