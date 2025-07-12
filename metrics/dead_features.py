import torch
from torchmetrics import Metric


class DeadFeatures(Metric):
    def __init__(
        self,
        n_features: int,
        n_layers: int,
        return_per_layer=False,
        return_neuron_indices=False,
        **kwargs,
    ):
        """
        :param n_features: number of features in the layer
        :param return_neuron_indices: whether to return the indices of the dead neurons
            if True, the compute method will return a tuple of (layer_indices, feature_indices)
            if False, the compute method will return the fraction of dead neurons
        """
        super().__init__(**kwargs)

        self.n_features = n_features
        self.n_layers = n_layers
        self.return_per_layer = return_per_layer
        self.return_neuron_indices = return_neuron_indices

        self.add_state(
            "dead_neurons",
            default=torch.zeros((n_layers, n_features), dtype=torch.float),
            dist_reduce_fx="sum",
        )

    def update(self, features: torch.Tensor):
        self.dead_neurons += features.detach().sum(dim=0)

    def compute(self):
        if self.return_neuron_indices:
            dead_mask = self.dead_neurons == 0.0
            layer_indices, feature_indices = dead_mask.nonzero(as_tuple=True)
            # Return tuple of (layer_indices, feature_indices) - more intuitive than flattened
            return (layer_indices, feature_indices)
        elif self.return_per_layer:
            return (self.dead_neurons == 0.0).float().mean(dim=1)
        return (self.dead_neurons == 0.0).float().mean()
