import os

# use cuda 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import nnsight
import torch
from einops import einsum

from model.clt import CrossLayerTranscoder
from utils.utils import get_webtext_dataloader


class ReplacementModel(torch.nn.Module):
    def __init__(self, gpt2, clt):
        super(ReplacementModel, self).__init__()
        self.gpt2 = gpt2
        self.clt = clt
        self.n_layers = self.gpt2.config.n_layer
        assert self.n_layers == self.clt.config["n_layers"]
        self.n_features = self.clt.config["d_features"]

    def forward(self, tokens):
        with self.gpt2.trace(tokens) as tracer:
            # features: batch_size x seq_len x n_layers x n_features
            features = torch.full(
                (tokens.shape[0], tokens.shape[1], self.n_layers, self.n_features),
                float("nan"),
            )

            for layer in range(self.n_layers):
                mlp_in = self.gpt2.transformer.h[
                    layer
                ].ln_2.input.save()  # batch_size x seq_len x d_resid
                # TODO: get features by running clt encoder on this layer
                mlp_in_norm = (mlp_in - mlp_in.mean(dim=-1, keepdim=True)) / mlp_in.std(
                    dim=-1, keepdim=True
                )
                features[..., layer, :] = einsum(
                    mlp_in_norm,
                    self.clt.W_enc[layer],
                    "batch seq d_acts, d_acts d_features -> batch seq d_features",
                )

                recons = einsum(
                    features[:layer],
                    self.clt.W_dec[:layer, layer],
                    "batch seq n_layers d_features, n_layers d_features d_acts -> batch seq d_acts",
                )

                mlp_out = self.gpt2.transformer.h[layer].mlp.output.save()
                # TODO: get reconstructions by running clt decoder on features of all lower layers
                self.gpt2.transformer.h[layer].mlp.output = recons
                logits = self.gpt2.lm_head.output.save()
        return logits


gpt2 = nnsight.LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
gpt2.requires_grad_(False)

loader = get_webtext_dataloader(gpt2)

clt = CrossLayerTranscoder.load_from_checkpoint("checkpoints/clt_model.ckpt")

replacement_model = ReplacementModel(gpt2, clt)

with torch.no_grad():
    for batch in loader:
        bos = torch.full(
            (batch.shape[0], 1),
            gpt2.config.bos_token_id,
            dtype=torch.long,
            device=batch.device,
        )
        batch = torch.cat([bos, batch], dim=1)
        logits_gpt2 = gpt2(batch.to(gpt2.device))
        print(logits_gpt2)
        logits_replacement = replacement_model(batch)
        print(logits_gpt2.shape, logits_replacement.shape)
        break
