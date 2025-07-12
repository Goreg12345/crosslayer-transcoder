import pytest
import torch

from metrics.dead_features import DeadFeatures


@pytest.fixture(
    params=[
        (10, 2),
        (5, 3),
        (100, 1),
        (768, 12),
        (1, 5),
    ],
    ids=["small", "medium", "single_layer", "transformer_like", "single_feature"],
)
def metric_params(request):
    """Creates test parameters for different metric configurations"""
    n_features, n_layers = request.param
    return n_features, n_layers


@pytest.fixture
def dead_features_metric(metric_params):
    """Creates a basic DeadFeatures metric"""
    n_features, n_layers = metric_params
    return DeadFeatures(n_features=n_features, n_layers=n_layers)


@pytest.fixture
def dead_features_per_layer(metric_params):
    """Creates a DeadFeatures metric with return_per_layer=True"""
    n_features, n_layers = metric_params
    return DeadFeatures(n_features=n_features, n_layers=n_layers, return_per_layer=True)


@pytest.fixture
def dead_features_neuron_indices(metric_params):
    """Creates a DeadFeatures metric with return_neuron_indices=True"""
    n_features, n_layers = metric_params
    return DeadFeatures(
        n_features=n_features, n_layers=n_layers, return_neuron_indices=True
    )


class TestDeadFeaturesBasic:
    """Test basic functionality of DeadFeatures metric"""

    def test_initialization(self, metric_params):
        """Test that the metric initializes correctly"""
        n_features, n_layers = metric_params
        metric = DeadFeatures(n_features=n_features, n_layers=n_layers)

        assert metric.n_features == n_features
        assert metric.n_layers == n_layers
        assert metric.return_per_layer == False
        assert metric.return_neuron_indices == False
        assert metric.dead_neurons.shape == (n_layers, n_features)
        assert torch.all(metric.dead_neurons == 0)

    def test_initialization_with_flags(self, metric_params):
        """Test initialization with different flags"""
        n_features, n_layers = metric_params
        metric = DeadFeatures(
            n_features=n_features,
            n_layers=n_layers,
            return_per_layer=True,
            return_neuron_indices=True,
        )

        assert metric.return_per_layer == True
        assert metric.return_neuron_indices == True


class TestDeadFeaturesZeroInput:
    """Test DeadFeatures with zero input (all neurons should be dead)"""

    def test_zeros_input_basic(self, dead_features_metric, metric_params):
        """Test with zeros input - should yield 1.0 (all dead)"""
        n_features, n_layers = metric_params

        # Create zeros tensor
        features = torch.zeros(10, n_layers, n_features)  # batch_size=10

        # Update metric
        dead_features_metric.update(features)

        # Compute result
        result = dead_features_metric.compute()

        # All features should be dead (never activated)
        assert torch.allclose(result, torch.tensor(1.0))

    def test_zeros_input_per_layer(self, dead_features_per_layer, metric_params):
        """Test with zeros input using return_per_layer=True"""
        n_features, n_layers = metric_params

        # Create zeros tensor
        features = torch.zeros(10, n_layers, n_features)  # batch_size=10

        # Update metric
        dead_features_per_layer.update(features)

        # Compute result
        result = dead_features_per_layer.compute()

        # Should return per-layer results, all 1.0 (all dead in each layer)
        expected = torch.ones(n_layers)
        assert torch.allclose(result, expected)

    def test_zeros_input_multiple_updates(self, dead_features_per_layer, metric_params):
        """Test with zeros input across multiple updates"""
        n_features, n_layers = metric_params

        # Create zeros tensors for multiple updates
        features1 = torch.zeros(5, n_layers, n_features)
        features2 = torch.zeros(8, n_layers, n_features)
        features3 = torch.zeros(3, n_layers, n_features)

        # Update metric multiple times
        dead_features_per_layer.update(features1)
        dead_features_per_layer.update(features2)
        dead_features_per_layer.update(features3)

        # Compute result
        result = dead_features_per_layer.compute()

        # Should still be all 1.0 (all dead in each layer)
        expected = torch.ones(n_layers)
        assert torch.allclose(result, expected)


class TestDeadFeaturesOnesInput:
    """Test DeadFeatures with ones input (all neurons should be alive)"""

    def test_ones_input_basic(self, dead_features_metric, metric_params):
        """Test with ones input - should yield 0.0 (all alive)"""
        n_features, n_layers = metric_params

        # Create ones tensor
        features = torch.ones(10, n_layers, n_features)  # batch_size=10

        # Update metric
        dead_features_metric.update(features)

        # Compute result
        result = dead_features_metric.compute()

        # All features should be alive (activated at least once)
        assert torch.allclose(result, torch.tensor(0.0))

    def test_ones_input_per_layer(self, dead_features_per_layer, metric_params):
        """Test with ones input using return_per_layer=True"""
        n_features, n_layers = metric_params

        # Create ones tensor
        features = torch.ones(10, n_layers, n_features)  # batch_size=10

        # Update metric
        dead_features_per_layer.update(features)

        # Compute result
        result = dead_features_per_layer.compute()

        # Should return per-layer results, all 0.0 (all alive in each layer)
        expected = torch.zeros(n_layers)
        assert torch.allclose(result, expected)

    def test_ones_input_multiple_updates(self, dead_features_per_layer, metric_params):
        """Test with ones input across multiple updates"""
        n_features, n_layers = metric_params

        # Create ones tensors for multiple updates
        features1 = torch.ones(5, n_layers, n_features)
        features2 = torch.ones(8, n_layers, n_features)
        features3 = torch.ones(3, n_layers, n_features)

        # Update metric multiple times
        dead_features_per_layer.update(features1)
        dead_features_per_layer.update(features2)
        dead_features_per_layer.update(features3)

        # Compute result
        result = dead_features_per_layer.compute()

        # Should still be all 0.0 (all alive in each layer)
        expected = torch.zeros(n_layers)
        assert torch.allclose(result, expected)


class TestDeadFeaturesIntermediate:
    """Test DeadFeatures with intermediate values"""

    def test_handpicked_intermediate_values(self):
        """Test with handpicked intermediate values"""
        n_features, n_layers = 4, 2
        metric = DeadFeatures(
            n_features=n_features, n_layers=n_layers, return_per_layer=True
        )

        # Create specific pattern: some features activate, others don't
        # Layer 0: [1, 0, 1, 0] - 2 dead out of 4 = 0.5
        # Layer 1: [0, 1, 0, 1] - 2 dead out of 4 = 0.5
        features = torch.tensor(
            [
                [
                    [1.0, 0.0, 1.0, 0.0],  # Layer 0
                    [0.0, 1.0, 0.0, 1.0],  # Layer 1
                ]
            ]
        )

        metric.update(features)
        result = metric.compute()

        expected = torch.tensor([0.5, 0.5])  # 50% dead in each layer
        assert torch.allclose(result, expected)

    def test_varying_activation_patterns(self):
        """Test with varying activation patterns across layers"""
        n_features, n_layers = 6, 3
        metric = DeadFeatures(
            n_features=n_features, n_layers=n_layers, return_per_layer=True
        )

        # Create specific patterns:
        # Layer 0: [1, 1, 0, 0, 0, 0] - 4 dead out of 6 = 4/6 ≈ 0.6667
        # Layer 1: [1, 0, 1, 0, 0, 0] - 4 dead out of 6 = 4/6 ≈ 0.6667
        # Layer 2: [0, 0, 0, 0, 0, 0] - 6 dead out of 6 = 1.0
        features = torch.tensor(
            [
                [
                    [1.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # Layer 0
                    [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],  # Layer 1
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Layer 2
                ]
            ]
        )

        metric.update(features)
        result = metric.compute()

        expected = torch.tensor([4.0 / 6.0, 4.0 / 6.0, 1.0])
        assert torch.allclose(result, expected)


class TestDeadFeaturesSpecificLayerActivation:
    """Test DeadFeatures with activation only in specific layers"""

    def test_single_layer_activation(self):
        """Test with ones only in one layer"""
        n_features, n_layers = 5, 4
        metric = DeadFeatures(
            n_features=n_features, n_layers=n_layers, return_per_layer=True
        )

        # Create pattern where only layer 1 has activations
        features = torch.zeros(10, n_layers, n_features)  # Start with zeros
        features[:, 1, :] = 5.0  # Activate all features in layer 1

        metric.update(features)
        result = metric.compute()

        # Layer 1 should have 0 dead features, others should have all dead
        expected = torch.tensor([1.0, 0.0, 1.0, 1.0])
        assert torch.allclose(result, expected)

    def test_partial_layer_activation(self):
        """Test with partial activation in specific layers"""
        n_features, n_layers = 6, 3
        metric = DeadFeatures(
            n_features=n_features, n_layers=n_layers, return_per_layer=True
        )

        # Create pattern where:
        # Layer 0: first 2 features activate
        # Layer 1: no activation
        # Layer 2: last 3 features activate
        features = torch.zeros(8, n_layers, n_features)
        features[:, 0, :2] = 1.0  # First 2 features in layer 0
        features[:, 2, -3:] = 1.0  # Last 3 features in layer 2

        metric.update(features)
        result = metric.compute()

        # Layer 0: 4 dead out of 6, Layer 1: 6 dead out of 6, Layer 2: 3 dead out of 6
        expected = torch.tensor([4.0 / 6.0, 1.0, 3.0 / 6.0])
        assert torch.allclose(result, expected)


class TestDeadFeaturesNeuronIndices:
    """Test DeadFeatures with return_neuron_indices=True"""

    def test_neuron_indices_basic(self):
        """Test neuron indices return functionality"""
        n_features, n_layers = 5, 2
        metric = DeadFeatures(
            n_features=n_features, n_layers=n_layers, return_neuron_indices=True
        )

        # Create pattern where features [0, 2] in layer 0 and [1, 3] in layer 1 are dead
        features = torch.zeros(5, n_layers, n_features)
        features[:, 0, [1, 3, 4]] = 1.0  # Activate features 1, 3, 4 in layer 0
        features[:, 1, [0, 2, 4]] = 1.0  # Activate features 0, 2, 4 in layer 1

        metric.update(features)
        layer_indices, feature_indices = metric.compute()

        # Should return tuple of (layer_indices, feature_indices)
        # Layer 0: dead features [0, 2] → layer_indices=[0, 0], feature_indices=[0, 2]
        # Layer 1: dead features [1, 3] → layer_indices=[1, 1], feature_indices=[1, 3]
        expected_layer_indices = torch.tensor([0, 0, 1, 1])
        expected_feature_indices = torch.tensor([0, 2, 1, 3])

        assert torch.equal(
            torch.sort(layer_indices)[0], torch.sort(expected_layer_indices)[0]
        )
        assert torch.equal(
            torch.sort(feature_indices)[0], torch.sort(expected_feature_indices)[0]
        )

    def test_neuron_indices_all_dead(self):
        """Test neuron indices when all neurons are dead"""
        n_features, n_layers = 3, 2
        metric = DeadFeatures(
            n_features=n_features, n_layers=n_layers, return_neuron_indices=True
        )

        # Don't update with any activations - all should be dead
        features = torch.zeros(1, n_layers, n_features)
        metric.update(features)
        layer_indices, feature_indices = metric.compute()

        # All indices should be returned
        expected_layer_indices = torch.tensor([0, 0, 0, 1, 1, 1])
        expected_feature_indices = torch.tensor([0, 1, 2, 0, 1, 2])

        assert torch.equal(
            torch.sort(layer_indices)[0], torch.sort(expected_layer_indices)[0]
        )
        assert torch.equal(
            torch.sort(feature_indices)[0], torch.sort(expected_feature_indices)[0]
        )

    def test_neuron_indices_none_dead(self):
        """Test neuron indices when no neurons are dead"""
        n_features, n_layers = 3, 2
        metric = DeadFeatures(
            n_features=n_features, n_layers=n_layers, return_neuron_indices=True
        )

        # Activate all neurons
        features = torch.ones(1, n_layers, n_features)
        metric.update(features)
        layer_indices, feature_indices = metric.compute()

        # No indices should be returned (empty tensors)
        assert layer_indices.numel() == 0
        assert feature_indices.numel() == 0


class TestDeadFeaturesEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_feature_single_layer(self):
        """Test with single feature and single layer"""
        metric = DeadFeatures(n_features=1, n_layers=1, return_per_layer=True)

        # Test with inactive feature
        features = torch.zeros(5, 1, 1)
        metric.update(features)
        result = metric.compute()
        assert torch.allclose(result, torch.tensor([1.0]))

        # Reset and test with active feature
        metric.reset()
        features = torch.ones(5, 1, 1)
        metric.update(features)
        result = metric.compute()
        assert torch.allclose(result, torch.tensor([0.0]))

    def test_reset_functionality(self):
        """Test that reset clears the metric state"""
        n_features, n_layers = 3, 2
        metric = DeadFeatures(
            n_features=n_features, n_layers=n_layers, return_per_layer=True
        )

        # Update with some activations
        features = torch.ones(5, n_layers, n_features)
        metric.update(features)

        # Should have no dead features
        result = metric.compute()
        assert torch.allclose(result, torch.zeros(n_layers))

        # Reset and check that all features are considered dead again
        metric.reset()
        result = metric.compute()
        assert torch.allclose(result, torch.ones(n_layers))

    def test_large_batch_consistency(self):
        """Test that results are consistent across different batch sizes"""
        n_features, n_layers = 10, 3
        metric1 = DeadFeatures(
            n_features=n_features, n_layers=n_layers, return_per_layer=True
        )
        metric2 = DeadFeatures(
            n_features=n_features, n_layers=n_layers, return_per_layer=True
        )

        # Generate same activations but in different batch sizes
        torch.manual_seed(42)
        features_large = torch.randn(100, n_layers, n_features)
        features_large = (features_large > 0.5).float()  # Convert to binary activations

        # Update first metric with large batch
        metric1.update(features_large)
        result1 = metric1.compute()

        # Update second metric with smaller batches
        for i in range(0, 100, 10):
            metric2.update(features_large[i : i + 10])
        result2 = metric2.compute()

        # Results should be identical
        assert torch.allclose(result1, result2)
