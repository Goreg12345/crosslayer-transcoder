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
            if True, the compute method will return a tensor with the indices of the dead neurons
            if False, the compute method will return the fraction of dead neurons
        """
        super().__init__(**kwargs)

        self.n_features = n_features
        self.n_layers = n_layers
        self.return_per_layer = return_per_layer
        self.return_neuron_indices = return_neuron_indices

        self.add_state(
            "dead_neurons",
            default=torch.zeros((n_layers, n_features), dtype=torch.int),
            dist_reduce_fx="sum",
        )

    def update(self, features: torch.Tensor):
        self.dead_neurons += (features > 0.0).sum(dim=0)

    def compute(self):
        if self.return_neuron_indices:
            return (self.dead_neurons == 0.0).nonzero(as_tuple=True)[0]
        elif self.return_per_layer:
            return (self.dead_neurons == 0.0).sum(dim=1) / self.n_features
        return (self.dead_neurons == 0.0).sum() / (self.n_layers * self.n_features)
