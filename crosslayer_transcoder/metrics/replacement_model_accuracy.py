import gc
from typing import Any

import nnsight
import torch
from torchmetrics import Metric

from crosslayer_transcoder.utils.utils import get_webtext_dataloader


class ReplacementModel(torch.nn.Module):
    def __init__(self, gpt2):
        super(ReplacementModel, self).__init__()
        self.gpt2 = gpt2
        self.n_layers = self.gpt2.config.n_layer

    def forward(self, tokens, clt):
        self.n_features = clt.encoder.d_features
        self.n_layers = clt.encoder.n_layers
        half = clt.encoder.W.dtype == torch.float16
        with self.gpt2.trace(tokens):
            # features: batch_size x seq_len x n_layers x n_features
            features = torch.full(
                (tokens.shape[0], tokens.shape[1], self.n_layers, self.n_features),
                float("nan"),
                device=tokens.device,
            )

            for layer in range(self.n_layers):
                mlp_in = self.gpt2.transformer.h[layer].ln_2.input  # (batch, seq, d_acts)

                mlp_in_norm = clt.input_standardizer(mlp_in, layer=layer).detach()

                if half:
                    mlp_in_norm = mlp_in_norm.to(torch.float16)
                pre_actvs = clt.encoder(mlp_in_norm, layer=layer)

                if clt.nonlinearity.__class__.__name__ == "JumpReLU":
                    feature_mask = torch.logical_and(
                        pre_actvs > clt.nonlinearity.theta[:, layer], pre_actvs > 0.0
                    )
                    features[..., layer, :] = feature_mask * pre_actvs
                elif isinstance(clt.nonlinearity, torch.nn.ReLU):
                    features[..., layer, :] = torch.relu(pre_actvs)
                else:
                    post_actvs = clt.nonlinearity(pre_actvs, layer=layer).detach()  # batchxseq, 1, n_features
                    features[..., layer, :] = post_actvs.reshape(
                        tokens.shape[0], tokens.shape[1], self.n_features
                    )

                if half:
                    features = features.to(torch.float16)
                recons = clt.decoder(features[..., : layer + 1, :], layer=layer)

                recons_norm = clt.output_standardizer(recons, layer=layer).detach()

                self.gpt2.transformer.h[layer].mlp.output = recons_norm
            logits = self.gpt2.lm_head.output.save()

        return logits


class ReplacementModelAccuracy(Metric):
    """
    Computes the accuracy of the replacement model and the KL divergence between the logits of the GPT-2 and the replacement model.
    """

    def __init__(self, model_name="openai-community/gpt2", device_map="auto", loader_batch_size=5):
        super().__init__()
        self.gpt2 = nnsight.LanguageModel(model_name, device_map=device_map, dispatch=True)
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

    def prepend_bos(self, tokens, mask):
        bos = torch.full(
            (tokens.shape[0], 1),
            self.gpt2.config.bos_token_id,
            dtype=torch.long,
            device=tokens.device,
        )
        tokens = torch.cat([bos, tokens], dim=1)
        mask = torch.cat(
            [torch.zeros((tokens.shape[0], 1), dtype=torch.bool, device=tokens.device), mask], dim=1
        )
        return tokens, mask

    def update(self, clt, max_batches=20):
        torch.cuda.empty_cache()
        gc.collect()
        with torch.no_grad():
            for i, (tokens, mask) in enumerate(self.loader):
                torch.cuda.empty_cache()
                print(f"computing replacement model", i)
                tokens = self.handle_device(tokens)
                mask = self.handle_device(mask)
                if i >= max_batches:
                    break
                tokens, mask = self.prepend_bos(tokens, mask)

                logits_gpt2 = self.gpt2(tokens)
                logits_replacement = self.replacement_model(tokens, clt)

                mask_flat = mask.reshape(-1)
                logits_gpt2 = logits_gpt2.logits
                logits_gpt2 = logits_gpt2.reshape(-1, logits_gpt2.shape[-1])[mask_flat]
                logits_replacement = logits_replacement.reshape(-1, logits_replacement.shape[-1])[mask_flat]

                self.n_correct += (
                    (logits_gpt2.argmax(dim=-1) == logits_replacement.argmax(dim=-1)).int().sum()
                )
                self.n_total += mask.sum()
                self.kl_div += torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(logits_gpt2, dim=-1),
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


class MoltReplacementModel(torch.nn.Module):
    """Run Gemma or GPT-2 with one MLP output replaced by a trained MOLT."""

    def __init__(self, language_model, model_arch: str, layer: int):
        super().__init__()
        if model_arch not in ("gpt2", "gemma3"):
            raise ValueError(f"Unsupported model_arch: {model_arch}")
        self.language_model = language_model
        self.model_arch = model_arch
        self.layer = layer
        self._gemma3_multimodal = None

    def _is_multimodal_gemma(self) -> bool:
        if self._gemma3_multimodal is None:
            model = self.language_model
            underlying = (
                getattr(model, "_model", None)
                or getattr(model, "local_model", None)
                or model
            )
            self._gemma3_multimodal = hasattr(underlying, "language_model")
        return self._gemma3_multimodal

    def _layer_module(self):
        if self.model_arch == "gpt2":
            return self.language_model.transformer.h[self.layer]
        if self._is_multimodal_gemma():
            return self.language_model.model.language_model.layers[self.layer]
        return self.language_model.model.layers[self.layer]

    def forward(self, tokens: torch.Tensor, molt) -> torch.Tensor:
        with self.language_model.trace(tokens):
            layer_module = self._layer_module()
            if self.model_arch == "gpt2":
                mlp_input = layer_module.ln_2.input
            else:
                mlp_input = layer_module.pre_feedforward_layernorm.input

            _, _, reconstruction = molt(mlp_input, self.layer)
            if self.model_arch == "gpt2":
                layer_module.mlp.output = reconstruction
            else:
                layer_module.post_feedforward_layernorm.output = reconstruction
            logits = self.language_model.lm_head.output.save()
        return logits


class MoltReplacementModelAccuracy(ReplacementModelAccuracy):
    """Agreement and KL metrics for a model with one layer replaced by MOLT."""

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        loader_batch_size: int = 2,
        model_arch: str = "gpt2",
        layer: int = 8,
        max_batches: int = 20,
        model_dtype: str = "bfloat16",
    ):
        Metric.__init__(self)
        self.language_model = nnsight.LanguageModel(
            model_name,
            device_map=device_map,
            dispatch=True,
            torch_dtype=getattr(torch, model_dtype),
        )
        self.language_model.requires_grad_(False)
        self.replacement_model = MoltReplacementModel(
            self.language_model, model_arch=model_arch, layer=layer
        )
        self.loader = get_webtext_dataloader(
            self.language_model, batch_size=loader_batch_size
        )
        self.max_batches = max_batches
        self.add_state("n_correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("n_total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state(
            "kl_div", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum"
        )
        self.add_state("n_kl_div", default=torch.tensor(0), dist_reduce_fx="sum")

    def _bos_token_id(self) -> int:
        config: Any = self.language_model.config
        bos_id = getattr(config, "bos_token_id", None)
        if bos_id is None and getattr(config, "text_config", None) is not None:
            bos_id = getattr(config.text_config, "bos_token_id", None)
        if bos_id is None:
            bos_id = getattr(self.language_model.tokenizer, "bos_token_id", None)
        if bos_id is None:
            raise RuntimeError("Could not resolve a BOS token id")
        return bos_id

    def update(self, molt):
        with torch.no_grad():
            for batch_index, (tokens, mask) in enumerate(self.loader):
                if batch_index >= self.max_batches:
                    break
                tokens = tokens.to(self.language_model.device)
                mask = mask.to(self.language_model.device)
                bos = torch.full(
                    (tokens.shape[0], 1),
                    self._bos_token_id(),
                    dtype=tokens.dtype,
                    device=tokens.device,
                )
                tokens = torch.cat([bos, tokens], dim=1)
                mask = torch.cat(
                    [
                        torch.zeros(
                            (mask.shape[0], 1), dtype=torch.bool, device=mask.device
                        ),
                        mask,
                    ],
                    dim=1,
                )

                reference_logits = self.language_model(tokens).logits
                replacement_logits = self.replacement_model(tokens, molt)
                flat_mask = mask.reshape(-1)
                reference_logits = reference_logits.reshape(
                    -1, reference_logits.shape[-1]
                )[flat_mask]
                replacement_logits = replacement_logits.reshape(
                    -1, replacement_logits.shape[-1]
                )[flat_mask]
                self.n_correct += (
                    reference_logits.argmax(-1) == replacement_logits.argmax(-1)
                ).sum()
                self.n_total += flat_mask.sum()
                self.kl_div += torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(reference_logits, dim=-1),
                    torch.nn.functional.log_softmax(replacement_logits, dim=-1),
                    reduction="batchmean",
                    log_target=True,
                )
                self.n_kl_div += 1
                self.language_model._clear()

    def compute(self):
        return self.n_correct / self.n_total, self.kl_div / self.n_kl_div
