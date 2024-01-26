from copy import deepcopy

import pytest
import timm
import torch

from source.utils.general import get_object_from_dict


@pytest.mark.parametrize(
    "backbone_name",
    [
        # For average deep computer vision enthusiast.
        *timm.list_models("*convnext*", pretrained=True),
        *timm.list_models("*efficientnet*", pretrained=True),
        # For average resnet enjoyer.
        *timm.list_models("*resnet*", pretrained=True),
    ],
)
def test_timm_simsiam(backbone_name, get_test_config, get_test_csv, get_test_extended_settings):
    is_local_run, enable_tests_on_gpu = get_test_extended_settings
    if is_local_run is None:
        pytest.skip("Skipping tests for bacbkones due to large amount of possible architectures")
    if "pruned" in backbone_name:
        pytest.skip("Skipping tests for pruned architectures")
    if backbone_name.startswith("tresnet"):
        pytest.skip("Skipping tests for TResNets due to the need of InplaceABN library")
    if "lambda" in backbone_name:
        pytest.skip("Skipping tests for LambdaNets due to fixed input size")
    config = deepcopy(get_test_config)
    model = get_object_from_dict(config.model)
    random_batch = torch.randn(config.training.batch_size, 3, 224, 224)

    model.eval()

    with torch.no_grad():
        # Check actual representation (tensor after avg. pooling).
        representation = model.get_representations(random_batch)
        assert representation.shape == (config.training.batch_size, model.embedding_dim)

        # Check full forward pass.
        out_dict = model(random_batch, random_batch)
        assert out_dict["view1_predictions"].shape == (config.training.batch_size, model.feature_dim)
        assert out_dict["view2_predictions"].shape == (config.training.batch_size, model.feature_dim)
        assert out_dict["view1_embeddings"].shape == (config.training.batch_size, model.feature_dim)
        assert out_dict["view2_embeddings"].shape == (config.training.batch_size, model.feature_dim)

    if enable_tests_on_gpu:
        random_batch = random_batch.cuda()
        model = model.cuda()
        random_batch.requires_grad = True
        out_dict = model(random_batch, random_batch)

        for param in model.parameters():
            assert param.grad is None

        assert out_dict["view1_predictions"].requires_grad
        assert out_dict["view2_predictions"].requires_grad
        assert not out_dict["view1_embeddings"].requires_grad
        assert not out_dict["view2_embeddings"].requires_grad

        # Let's calculate dumb loss.
        loss1 = -(out_dict["view1_predictions"] * out_dict["view1_embeddings"]).sum(dim=1).mean()
        loss2 = -(out_dict["view2_predictions"] * out_dict["view2_embeddings"]).sum(dim=1).mean()
        loss = 0.5 * (loss1 + loss2)
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None
        assert random_batch.grad is not None
