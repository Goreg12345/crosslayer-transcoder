import torch
from torchmetrics import Metric


class DeadFeatures(Metric):
    def __init__(
        self,
        n_features: int,
        n_layers: int,
        return_per_layer=False,
        return_neuron_indices=False,
        return_log_freqs=False,
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
        self.return_log_freqs = return_log_freqs

        self.add_state(
            "n_active",
            default=torch.zeros((n_layers, n_features), dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "n_total",
            default=torch.zeros((1), dtype=torch.long),
            dist_reduce_fx="sum",
        )

    def update(self, features: torch.Tensor):
        self.n_active += (features.detach() > 0.0).sum(dim=0)
        self.n_total += features.shape[0]

    def compute(self):
        return_dict = {}
        return_dict["mean"] = (self.n_active == 0.0).float().mean()
        if self.return_neuron_indices:
            dead_mask = self.n_active == 0.0
            layer_indices, feature_indices = dead_mask.nonzero(as_tuple=True)
            # Return tuple of (layer_indices, feature_indices) - more intuitive than flattened
            return_dict["layer_indices"] = layer_indices
            return_dict["feature_indices"] = feature_indices
        elif self.return_per_layer:
            return_dict["per_layer"] = (self.n_active == 0.0).float().mean(dim=1)
        if self.return_log_freqs:
            return_dict["log_freqs"] = torch.clamp(
                torch.log10(self.n_active / self.n_total), min=-10
            )
        return return_dict
