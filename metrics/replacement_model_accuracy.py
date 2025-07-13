import gc

import nnsight
import torch
from einops import einsum
from torchmetrics import Metric

from model.jumprelu import JumpReLU
from utils.utils import get_webtext_dataloader


class ReplacementModel(torch.nn.Module):
    def __init__(self, gpt2):
        super(ReplacementModel, self).__init__()
        self.gpt2 = gpt2
        self.n_layers = self.gpt2.config.n_layer

    def forward(self, tokens, clt):
        self.n_features = clt.d_features
        self.n_layers = clt.n_layers
        half = clt.W_enc.dtype == torch.float16
        with self.gpt2.trace(tokens):
            # features: batch_size x seq_len x n_layers x n_features
            features = torch.full(
                (tokens.shape[0], tokens.shape[1], self.n_layers, self.n_features),
                float("nan"),
                device=tokens.device,
            )

            for layer in range(self.n_layers):
                mlp_in = self.gpt2.transformer.h[
                    layer
                ].ln_2.input  # (batch, seq, d_acts)

                mlp_in_norm = clt.input_standardizer(mlp_in, layer=layer).detach()

                if half:
                    mlp_in_norm = mlp_in_norm.to(torch.float16)
                pre_actvs = einsum(
                    mlp_in_norm,
                    clt.W_enc[layer].detach(),
                    "batch seq d_acts, d_acts d_features -> batch seq d_features",
                )

                if isinstance(clt.nonlinearity, JumpReLU):
                    feature_mask = torch.logical_and(
                        pre_actvs > clt.nonlinearity.theta[:, layer], pre_actvs > 0.0
                    )
                    features[..., layer, :] = feature_mask * pre_actvs
                else:
                    post_actvs = clt.nonlinearity(
                        pre_actvs, training=False
                    ).detach()  # batchxseq, 1, n_features
                    features[..., layer, :] = post_actvs.reshape(
                        tokens.shape[0], tokens.shape[1], self.n_features
                    )

                if half:
                    features = features.to(torch.float16)
                recons = einsum(
                    features[..., : layer + 1, :],
                    clt.W_dec[: layer + 1, layer].detach(),
                    "batch seq n_layers d_features, n_layers d_features d_acts -> batch seq d_acts",
                )

                recons_norm = clt.output_standardizer(recons, layer=layer).detach()

                self.gpt2.transformer.h[layer].mlp.output = recons_norm
            logits = self.gpt2.lm_head.output.save()

        return logits


class ReplacementModelAccuracy(Metric):
    """
    Computes the accuracy of the replacement model and the KL divergence between the logits of the GPT-2 and the replacement model.
    """

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
        self.add_state(
            "kl_div",
            default=torch.tensor(0, dtype=torch.float32),
            dist_reduce_fx="sum",
        )
        self.add_state("n_kl_div", default=torch.tensor(0), dist_reduce_fx="sum")

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
        torch.cuda.empty_cache()
        gc.collect()
        with torch.no_grad():
            for i, tokens in enumerate(self.loader):
                torch.cuda.empty_cache()
                print(f"computing replacement model", i)
                tokens = self.handle_device(tokens)
                if i >= max_batches:
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
                self.kl_div += torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(logits_gpt2.logits, dim=-1),
                    torch.nn.functional.log_softmax(logits_replacement, dim=-1),
                    reduction="batchmean",
                    log_target=True,
                )
                self.n_kl_div += 1
                del logits_gpt2, logits_replacement
                # gc.collect()
                self.gpt2._clear()
                self.replacement_model.gpt2._clear()
                torch.cuda.empty_cache()
                gc.collect()
        self.gpt2._clear()
        self.replacement_model.gpt2._clear()
        print("exiting update")
        gc.collect()
        torch.cuda.empty_cache()

    def compute(self):
        return self.n_correct / self.n_total, self.kl_div / self.n_kl_div
