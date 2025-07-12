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
    """Creates test tensors with various shapes - all positive values"""
    shape = request.param
    return torch.rand(*shape) + 0.1  # Ensures all values are positive (0.1 to 1.1)


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


def test_no_negative_values_in_output(per_layer_topk, per_sample_topk, batch_topk):
    """Test that all topk implementations never output negative values, even with negative inputs"""
    # Create test tensor with negative values
    negative_features = torch.randn(2, 3, 4) - 1.0  # Ensures mostly negative values

    # Test PerLayerTopK
    result = per_layer_topk(negative_features)
    assert torch.all(result >= 0), "PerLayerTopK output contains negative values"

    # Test PerSampleTopK
    result = per_sample_topk(negative_features)
    assert torch.all(result >= 0), "PerSampleTopK output contains negative values"

    # Test BatchTopK
    result = batch_topk(negative_features)
    assert torch.all(result >= 0), "BatchTopK output contains negative values"


def test_mixed_positive_negative_values(per_layer_topk, per_sample_topk, batch_topk):
    """Test that topk implementations handle mixed positive/negative values correctly"""
    # Create test tensor with mix of positive and negative values
    mixed_features = torch.tensor(
        [
            [
                [1.0, -2.0, 3.0, -4.0, 5.0],  # Layer 0: mix of pos/neg
                [-1.0, 2.0, -3.0, 4.0, -5.0],  # Layer 1: mix of pos/neg
            ]
        ]
    )

    # Test PerLayerTopK - should keep top 3 positive values per layer
    result = per_layer_topk(mixed_features)
    assert torch.all(result >= 0), "PerLayerTopK output contains negative values"
    # For layer 0: should keep [1.0, 3.0, 5.0] (top 3 values, negatives become 0)
    # For layer 1: should keep [2.0, 4.0] (top 2 positive values, 3rd would be negative so becomes 0)

    # Test PerSampleTopK - should keep top 3 values across all layers
    result = per_sample_topk(mixed_features)
    assert torch.all(result >= 0), "PerSampleTopK output contains negative values"

    # Test BatchTopK - should keep top 3 values across entire batch
    result = batch_topk(mixed_features)
    assert torch.all(result >= 0), "BatchTopK output contains negative values"


@pytest.fixture
def nnsight_model():
    import nnsight

    gpt2 = nnsight.LanguageModel(
        "openai-community/gpt2", device_map="auto", dispatch=True
    )

    gpt2.requires_grad_(False)
    return gpt2


@pytest.mark.parametrize(
    "topk_fixture",
    ["per_layer_topk", "per_sample_topk", "batch_topk"],
)
def test_nnsight_compatibility(nnsight_model, topk_fixture, request):
    """Test that all topk implementations work correctly with nnsight tracing"""
    topk = request.getfixturevalue(topk_fixture)
    topk.to(nnsight_model.device)

    # Test that no error occurs during tracing and topk execution
    try:
        with nnsight_model.trace("test"):
            actvs = nnsight_model.transformer.h[0].mlp.output
            topk.to(actvs.device)
            topk_result = topk(actvs)
    except Exception as e:
        pytest.fail(f"nnsight compatibility test failed for {topk_fixture}: {e}")
