import pytest
import torch

from crosslayer_transcoder.model.topk import BatchTopK, PerLayerBatchTopK, PerLayerTopK


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
def per_layer_batch_topk(features):
    # Get the actual number of layers from the test data
    _, n_layers, _ = features.shape
    return PerLayerBatchTopK(k=3, e=0.1, n_layers=n_layers)


@pytest.fixture
def batch_topk():
    return BatchTopK(k=3, e=0.1)


def test_per_layer_topk_output_shape(per_layer_topk, features):
    topk_features = per_layer_topk(features)
    assert topk_features.shape == features.shape


def test_per_layer_topk_has_k_nonzero_values(per_layer_topk, features):
    topk_features = per_layer_topk(features)
    assert (topk_features[0, 0] != 0).sum() == per_layer_topk.k


def test_per_layer_batch_topk_output_shape(per_layer_batch_topk, features):
    topk_features = per_layer_batch_topk(features)
    assert topk_features.shape == features.shape


def test_per_layer_batch_topk_has_k_nonzero_values(per_layer_batch_topk, features):
    batch_size, n_layers, d_features = features.shape
    topk_features = per_layer_batch_topk(features)
    # Each layer should have k * batch_size non-zero values
    for layer in range(n_layers):
        layer_nonzeros = (topk_features[:, layer, :] != 0).sum()
        assert layer_nonzeros == per_layer_batch_topk.k * batch_size


def test_batch_topk_output_shape(batch_topk, features):
    topk_features = batch_topk(features)
    assert topk_features.shape == features.shape


def test_batch_topk_has_k_nonzero_values(batch_topk, features):
    batch_size, n_layers, d_features = features.shape
    topk_features = batch_topk(features)
    # Should have k * batch_size * n_layers non-zero values total
    expected_nonzeros = batch_topk.k * batch_size * n_layers
    assert (topk_features != 0).sum() == expected_nonzeros


def test_per_layer_topk_has_correct_values(per_layer_topk):
    features = torch.tensor([[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]])
    topk_features = per_layer_topk(features)
    assert torch.allclose(topk_features[0, 0], torch.tensor([0, 0, 3, 4, 5]))
    assert torch.allclose(topk_features[0, 1], torch.tensor([0, 0, 8, 9, 10]))


def test_per_layer_batch_topk_has_correct_values():
    per_layer_batch_topk = PerLayerBatchTopK(k=3, e=0.1, n_layers=2)
    features = torch.tensor(
        [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]]]
    )
    topk_features = per_layer_batch_topk(features)
    # Layer 0: top 6 values from [1,2,3,4,5,11,12,13,14,15] -> keep [11,12,13,14,15,5]
    # Layer 1: top 6 values from [6,7,8,9,10,16,17,18,19,20] -> keep [16,17,18,19,20,10]
    assert (topk_features[:, 0, :] != 0).sum() == 6  # 3 * 2 batch_size
    assert (topk_features[:, 1, :] != 0).sum() == 6  # 3 * 2 batch_size


def test_batch_topk_has_correct_values():
    batch_topk = BatchTopK(k=3, e=0.1)
    features = torch.tensor(
        [
            [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
        ]
    )
    topk_features = batch_topk(features)
    # Should keep top 12 values globally (3 * 2 batch * 2 layers)
    # Top 12: [20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9]
    assert (topk_features != 0).sum() == 12
    # Should contain the highest values
    assert topk_features[1, 1, 4] == 20  # Highest value should be preserved
    assert topk_features[1, 1, 3] == 19  # Second highest should be preserved


def test_per_layer_topk_backward(per_layer_topk, features):
    features.requires_grad = True
    topk_features = per_layer_topk(features)
    topk_features.sum().backward()
    assert features.grad is not None


def test_per_layer_batch_topk_backward(per_layer_batch_topk, features):
    features.requires_grad = True
    topk_features = per_layer_batch_topk(features)
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

    topk = PerLayerBatchTopK(k=0, e=0.1, n_layers=features.shape[1])
    result = topk(features)
    assert torch.allclose(result, torch.zeros_like(features))

    topk = BatchTopK(k=0, e=0.1)
    result = topk(features)
    assert torch.allclose(result, torch.zeros_like(features))


def test_k_boundary(features):
    batch_size, n_layers, d_features = features.shape

    topk = PerLayerTopK(k=d_features)
    result = topk(features)
    assert torch.allclose(result, features)

    topk = PerLayerBatchTopK(k=d_features, e=0.1, n_layers=n_layers)
    result = topk(features)
    assert torch.allclose(result, features)

    topk = BatchTopK(k=d_features, e=0.1)
    result = topk(features)
    assert torch.allclose(result, features)


def test_k_boundary_plus_one_error(features):
    batch_size, n_layers, d_features = features.shape

    topk = PerLayerTopK(k=d_features + 1)
    with pytest.raises(AssertionError):
        topk(features)

    topk = PerLayerBatchTopK(k=d_features + 1, e=0.1, n_layers=n_layers)
    with pytest.raises(AssertionError):
        topk(features)

    topk = BatchTopK(k=d_features + 1, e=0.1)
    with pytest.raises(AssertionError):
        topk(features)


def test_no_negative_values_in_output(per_layer_topk, per_layer_batch_topk, batch_topk):
    """Test that all topk implementations never output negative values, even with negative inputs"""
    # Create test tensor with negative values
    negative_features = torch.randn(2, 3, 4) - 1.0  # Ensures mostly negative values

    # Test PerLayerTopK
    result = per_layer_topk(negative_features)
    assert torch.all(result >= 0), "PerLayerTopK output contains negative values"

    # Test PerLayerBatchTopK
    result = per_layer_batch_topk(negative_features)
    assert torch.all(result >= 0), "PerLayerBatchTopK output contains negative values"

    # Test BatchTopK
    result = batch_topk(negative_features)
    assert torch.all(result >= 0), "BatchTopK output contains negative values"


def test_mixed_positive_negative_values(per_layer_topk, per_layer_batch_topk, batch_topk):
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

    # Test PerLayerBatchTopK - should keep top 3 values per layer across batch
    result = per_layer_batch_topk(mixed_features)
    assert torch.all(result >= 0), "PerLayerBatchTopK output contains negative values"

    # Test BatchTopK - should keep top 3 values across entire batch
    result = batch_topk(mixed_features)
    assert torch.all(result >= 0), "BatchTopK output contains negative values"


@pytest.fixture
def nnsight_model():
    import nnsight

    gpt2 = nnsight.LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

    gpt2.requires_grad_(False)
    return gpt2


@pytest.mark.parametrize(
    "topk_fixture",
    ["per_layer_topk", "per_layer_batch_topk", "batch_topk"],
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
            topk(actvs)
    except Exception as e:
        pytest.fail(f"nnsight compatibility test failed for {topk_fixture}: {e}")
