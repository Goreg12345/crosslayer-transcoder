"""Load only the encoder, standardization, and gate thresholds from a MOLT checkpoint."""

from torch import nn

from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.standardize import DimensionwiseInputStandardizer


class MoltGates(nn.Module):
    def __init__(self, state, layer):
        super().__init__()
        weight = state["model.e.weight"]
        self.layer = layer
        self.encoder = nn.Linear(weight.shape[1], weight.shape[0])
        self.encoder.load_state_dict({"weight": weight, "bias": state["model.e.bias"]})
        mean = state["model.input_standardizer.mean"]
        self.standardizer = DimensionwiseInputStandardizer(*mean.shape).to(mean.dtype)
        self.standardizer.load_state_dict(
            {k: state[f"model.input_standardizer.{k}"] for k in ("mean", "std")}
        )
        self.standardizer.is_initialized = True
        self.gate = JumpReLU(n_layers=1, d_features=weight.shape[0])
        self.gate.load_state_dict(
            {k: state[f"model.nonlinearity.{k}"] for k in ("theta", "bandwidth")}
        )
        self.requires_grad_(False)

    def forward(self, x):
        return self.gate(self.encoder(self.standardizer(x, self.layer)))
