import nnsight
import torch
from einops import einsum
from torchmetrics import Metric

from utils.utils import get_webtext_dataloader


class ReplacementModel(torch.nn.Module):
    def __init__(self, gpt2):
        super(ReplacementModel, self).__init__()
        self.gpt2 = gpt2
        self.n_layers = self.gpt2.config.n_layer

    def forward(self, tokens, clt):
        self.n_features = clt.d_features
        self.n_layers = clt.n_layers
        with self.gpt2.trace(tokens):
            # features: batch_size x seq_len x n_layers x n_features
            features = torch.full(
                (tokens.shape[0], tokens.shape[1], self.n_layers, self.n_features),
                float("nan"),
                device=tokens.device,
            )

            for layer in range(self.n_layers):
                mlp_in = self.gpt2.transformer.h[layer].ln_2.input

                mean = mlp_in.mean(dim=-1, keepdim=True)
                std = mlp_in.std(dim=-1, keepdim=True)
                mlp_in_norm = (mlp_in - mean) / std

                pre_actvs = einsum(
                    mlp_in_norm,
                    clt.W_enc[layer],
                    "batch seq d_acts, d_acts d_features -> batch seq d_features",
                )

                feature_mask = torch.logical_and(
                    pre_actvs > clt.nonlinearity.theta[:, layer], pre_actvs > 0.0
                )
                features[..., layer, :] = feature_mask * pre_actvs

                recons = einsum(
                    features[..., : layer + 1, :],
                    clt.W_dec[: layer + 1, layer],
                    "batch seq n_layers d_features, n_layers d_features d_acts -> batch seq d_acts",
                )

                self.gpt2.transformer.h[layer].mlp.output = recons
            logits = self.gpt2.lm_head.output.save()

        return logits


class ReplacementModelAccuracy(Metric):
    def __init__(
        self, model_name="openai-community/gpt2", device_map="auto", loader_batch_size=5
    ):
        super().__init__()
        self.gpt2 = nnsight.LanguageModel(
            model_name, device_map=device_map, dispatch=True
        )
        self.gpt2.requires_grad_(False)
        self.replacement_model = ReplacementModel(self.gpt2)
        self.loader = get_webtext_dataloader(self.gpt2, batch_size=loader_batch_size)
        self.add_state("n_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_total", default=torch.tensor(0), dist_reduce_fx="sum")

    def handle_device(self, tokens):
        tokens = tokens.to(self.gpt2.device)
        return tokens

    def prepend_bos(self, tokens):
        bos = torch.full(
            (tokens.shape[0], 1),
            self.gpt2.config.bos_token_id,
            dtype=torch.long,
            device=tokens.device,
        )
        tokens = torch.cat([bos, tokens], dim=1)
        return tokens

    def update(self, clt, max_batches=10):
        with torch.no_grad():
            for i, tokens in enumerate(self.loader):
                print(f"computing replacement model", i)
                tokens = self.handle_device(tokens)
                if i > max_batches:
                    break
                tokens = self.prepend_bos(tokens)

                logits_gpt2 = self.gpt2(tokens)
                logits_replacement = self.replacement_model(tokens, clt)

                self.n_correct += (
                    (
                        logits_gpt2.logits.argmax(dim=-1)
                        == logits_replacement.argmax(dim=-1)
                    )
                    .int()
                    .sum()
                )
                self.n_total += tokens.numel()

        print("exiting update")

    def compute(self):
        return self.n_correct / self.n_total
