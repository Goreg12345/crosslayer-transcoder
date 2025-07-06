import pytest
import torch

from model.topk import BatchTopK, PerLayerTopK, PerSampleTopK


@pytest.fixture(
    params=[
        (3, 4, 5),
        (1, 1, 5),
        (2, 8, 10),
        (1, 12, 768),
        (10, 1, 3),
    ],
    ids=["standard", "minimal", "medium", "transformer_like", "single_layer"],
)
def features(request):
    """Creates test tensors with various shapes"""
    shape = request.param
    return torch.randn(*shape)


@pytest.fixture
def per_layer_topk():
    return PerLayerTopK(k=3)


@pytest.fixture
def per_sample_topk():
    return PerSampleTopK(k=3)


@pytest.fixture
def batch_topk():
    return BatchTopK(k=3)


def test_per_layer_topk_output_shape(per_layer_topk, features):
    topk_features = per_layer_topk(features)
    assert topk_features.shape == features.shape


def test_per_layer_topk_has_k_nonzero_values(per_layer_topk, features):
    topk_features = per_layer_topk(features)
    assert (topk_features[0, 0] != 0).sum() == per_layer_topk.k


def test_per_sample_topk_output_shape(per_sample_topk, features):
    topk_features = per_sample_topk(features)
    assert topk_features.shape == features.shape


def test_per_sample_topk_has_k_nonzero_values(per_sample_topk, features):
    topk_features = per_sample_topk(features)
    assert (topk_features[0] != 0).sum() == per_sample_topk.k


def test_batch_topk_output_shape(batch_topk, features):
    topk_features = batch_topk(features)
    assert topk_features.shape == features.shape


def test_batch_topk_has_k_nonzero_values(batch_topk, features):
    topk_features = batch_topk(features)
    assert (topk_features != 0).sum() == batch_topk.k


def test_per_layer_topk_has_correct_values(per_layer_topk):
    features = torch.tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]])
    topk_features = per_layer_topk(features)
    assert torch.allclose(topk_features[0, 0], torch.tensor([0, 0, 3, 4, 5]))
    assert torch.allclose(topk_features[0, 1], torch.tensor([0, 0, 8, 9, 10]))


def test_per_sample_topk_has_correct_values(per_sample_topk):
    features = torch.tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]])
    topk_features = per_sample_topk(features)
    assert torch.allclose(topk_features[0, 0], torch.tensor([0, 0, 0, 0, 0]))
    assert torch.allclose(topk_features[0, 1], torch.tensor([0, 0, 8, 9, 10]))


def test_batch_topk_has_correct_values(batch_topk):
    features = torch.tensor(
        [
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
        ]
    )
    topk_features = batch_topk(features)
    assert torch.allclose(topk_features[0, 0], torch.tensor([0, 0, 0, 0, 0]))
    assert torch.allclose(topk_features[0, 1], torch.tensor([0, 0, 0, 0, 0]))
    assert torch.allclose(topk_features[1, 0], torch.tensor([0, 0, 0, 0, 0]))
    assert torch.allclose(topk_features[1, 1], torch.tensor([0, 0, 18, 19, 20]))


def test_per_layer_topk_backward(per_layer_topk, features):
    features.requires_grad = True
    topk_features = per_layer_topk(features)
    topk_features.sum().backward()
    assert features.grad is not None


def test_per_sample_topk_backward(per_sample_topk, features):
    features.requires_grad = True
    topk_features = per_sample_topk(features)
    topk_features.sum().backward()
    assert features.grad is not None


def test_batch_topk_backward(batch_topk, features):
    features.requires_grad = True
    topk_features = batch_topk(features)
    topk_features.sum().backward()
    assert features.grad is not None


def test_k_zero(features):
    topk = PerLayerTopK(k=0)
    result = topk(features)
    assert torch.allclose(result, torch.zeros_like(features))

    topk = PerSampleTopK(k=0)
    result = topk(features)
    assert torch.allclose(result, torch.zeros_like(features))

    topk = BatchTopK(k=0)
    result = topk(features)
    assert torch.allclose(result, torch.zeros_like(features))


def test_k_boundary(features):
    batch_size, n_layers, d_features = features.shape

    topk = PerLayerTopK(k=d_features)
    result = topk(features)
    assert torch.allclose(result, features)

    topk = PerSampleTopK(k=d_features * n_layers)
    result = topk(features)
    assert torch.allclose(result, features)

    topk = BatchTopK(k=features.numel())
    result = topk(features)
    assert torch.allclose(result, features)


def test_k_boundary_plus_one_error(features):
    batch_size, n_layers, d_features = features.shape

    topk = PerLayerTopK(k=d_features + 1)
    with pytest.raises(AssertionError):
        topk(features)

    topk = PerSampleTopK(k=d_features * n_layers + 1)
    with pytest.raises(AssertionError):
        topk(features)

    topk = BatchTopK(k=features.numel() + 1)
    with pytest.raises(AssertionError):
        topk(features)
