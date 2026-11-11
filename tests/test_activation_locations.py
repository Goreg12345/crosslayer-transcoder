"""Compare configurable extraction with hooks on tiny, randomly initialized LMs."""

import lightning as L
import nnsight
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    GPT2Config, GPT2LMHeadModel, Qwen3Config, Qwen3ForCausalLM,
    Gemma3TextConfig, Gemma3ForCausalLM, PreTrainedTokenizerFast,
)

from crosslayer_transcoder.data.activation_sources import ActivationComputer
from crosslayer_transcoder.data.datamodule import ActivationDataModule
from crosslayer_transcoder.model.clt import CrossLayerTranscoder, Encoder, Decoder
from crosslayer_transcoder.model.clt_lightning import TopKCrossLayerTranscoderModule
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer, DimensionwiseOutputStandardizer,
)
from crosslayer_transcoder.model.topk import PerLayerBatchTopK, PerLayerTopK


@pytest.fixture(scope="module", params=["gpt2", "qwen3", "gemma3"])
def tiny_model(request):
    torch.manual_seed(0)
    arch = request.param
    if arch == "gpt2":
        raw = GPT2LMHeadModel(GPT2Config(vocab_size=32, n_embd=16, n_layer=2, n_head=4))
        layers = raw.transformer.h
    else:
        config_cls, model_cls = (
            (Qwen3Config, Qwen3ForCausalLM) if arch == "qwen3"
            else (Gemma3TextConfig, Gemma3ForCausalLM)
        )
        raw = model_cls(config_cls(
            vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
            num_attention_heads=4, num_key_value_heads=2, head_dim=4,
        ))
        layers = raw.model.layers
    raw.eval()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1}, unk_token="[UNK]")),
        pad_token="[PAD]", unk_token="[UNK]",
    )
    return arch, raw, layers, nnsight.LanguageModel(raw, tokenizer=tokenizer)


TOKENS = torch.tensor([[1, 2, 3], [4, 5, 0]])
MASK = torch.tensor([[True, True, True], [True, True, False]])


@pytest.mark.parametrize("location", ["query", "key", "value", "head_output"])
@pytest.mark.parametrize("selected_layer", [None, 1])
def test_heads_match_real_projection_hooks(tiny_model, location, selected_layer):
    arch, raw, layers, model = tiny_model
    captured, hooks = {}, []
    selected = range(2) if selected_layer is None else [selected_layer]
    for i in selected:
        attention = layers[i].attn if arch == "gpt2" else layers[i].self_attn
        if location == "head_output":
            projection = attention.c_proj if arch == "gpt2" else attention.o_proj
            hooks.append(projection.register_forward_pre_hook(
                lambda module, args, i=i: captured.update({i: args[0].detach().clone()})
            ))
        else:
            projection = attention.c_attn if arch == "gpt2" else getattr(
                attention, {"query": "q_proj", "key": "k_proj", "value": "v_proj"}[location]
            )
            hooks.append(projection.register_forward_hook(
                lambda module, args, output, i=i: captured.update({i: output.detach().clone()})
            ))
    try:
        with torch.no_grad():
            raw(TOKENS)
    finally:
        for hook in hooks:
            hook.remove()
    if arch == "gpt2" and location != "head_output":
        offset = {"query": 0, "key": 16, "value": 32}[location]
        captured = {i: value[..., offset:offset + 16] for i, value in captured.items()}
    heads = captured[next(iter(selected))].shape[-1] // 4
    slots = len(selected) * heads
    expected = torch.stack([
        captured[i][..., head * 4:(head + 1) * 4][MASK]
        for i in selected for head in range(heads)
    ], dim=1)
    # Exercise DataModule validation too: n_layers now counts per-head dictionaries.
    data = ActivationDataModule(
        n_layers=slots, activation_layer=selected_layer, activation_dim=4,
        activation_input_location=location, activation_output_location=location,
    )
    actual = ActivationComputer(
        data.n_layers, model_arch=arch, input_location=location,
        output_location=location, layer=data.activation_layer,
    ).get_next_batch(model, TOKENS, MASK)
    assert actual.shape == (5, 2, slots, 4)
    torch.testing.assert_close(actual[:, 0], expected)
    torch.testing.assert_close(actual[:, 1], expected)


def test_explicit_same_location_and_legacy_defaults(tiny_model):
    arch, raw, layers, model = tiny_model
    captured = []
    hook = layers[1].mlp.register_forward_hook(
        lambda module, args, output: captured.append((args[0].detach().clone(), output.detach().clone()))
    )
    try:
        with torch.no_grad():
            raw(TOKENS)
    finally:
        hook.remove()
    actual = ActivationComputer(
        1, model_arch=arch, input_location="mlp.output", output_location="mlp.output", layer=1,
    ).get_next_batch(model, TOKENS, MASK)
    torch.testing.assert_close(actual[:, 0, 0], captured[0][1][MASK])
    torch.testing.assert_close(actual[:, 0], actual[:, 1])
    paired = ActivationComputer(
        1, model_arch=arch, input_location="mlp.input", output_location="mlp.output", layer=1,
    ).get_next_batch(model, TOKENS, MASK)
    torch.testing.assert_close(paired[:, 0, 0], captured[0][0][MASK])
    torch.testing.assert_close(paired[:, 1, 0], captured[0][1][MASK])
    full = ActivationComputer(2, model_arch=arch).get_next_batch(model, TOKENS, MASK)
    selected = ActivationComputer(1, model_arch=arch, layer=1).get_next_batch(model, TOKENS, MASK)
    torch.testing.assert_close(selected, full[:, :, 1:2])
    assert not torch.equal(selected[:, 0], selected[:, 1])


def test_block_input_and_tuple_output_locations(tiny_model):
    arch, raw, layers, model = tiny_model
    captured = []

    def capture(module, args, output):
        value = output[0] if isinstance(output, tuple) else output
        captured.append((args[0].detach().clone(), value.detach().clone()))

    hook = layers[1].register_forward_hook(capture)
    try:
        with torch.no_grad():
            raw(TOKENS)
    finally:
        hook.remove()
    actual = ActivationComputer(
        1, model_arch=arch, input_location="input", output_location="output", layer=1,
    ).get_next_batch(model, TOKENS, MASK)
    for side in (0, 1):
        torch.testing.assert_close(actual[:, side, 0], captured[0][side][MASK])


def test_mismatched_head_counts_and_slot_counts_fail(tiny_model):
    arch, _, _, model = tiny_model
    with pytest.raises(ValueError, match="n_layers=1 slots"):
        ActivationComputer(1, model_arch=arch, input_location="query", output_location="query").get_next_batch(
            model, TOKENS, MASK
        )
    if arch != "gpt2":
        with pytest.raises(ValueError, match="head counts may differ"):
            ActivationComputer(8, model_arch=arch, input_location="query", output_location="key").get_next_batch(
                model, TOKENS, MASK
            )


def test_per_head_sae_trains_with_independent_weights(tiny_model, tmp_path):
    arch, _, _, model = tiny_model
    batch = ActivationComputer(8, model_arch=arch, input_location="query", output_location="query").get_next_batch(
        model, TOKENS, MASK
    )
    sae = CrossLayerTranscoder(
        encoder=Encoder(d_acts=4, d_features=8, n_layers=8),
        decoder=Decoder(d_acts=4, d_features=8, n_layers=8),
        nonlinearity=PerLayerBatchTopK(k=2, e=0.01, n_layers=8),
        input_standardizer=DimensionwiseInputStandardizer(n_layers=8, activation_dim=4),
        output_standardizer=DimensionwiseOutputStandardizer(n_layers=8, activation_dim=4),
    )
    module = TopKCrossLayerTranscoderModule(
        model=sae, topk_aux=PerLayerTopK(k=4), compile=False,
        compute_dead_features=False, log_metrics_every=1,
    )
    trainer = L.Trainer(
        max_steps=2, accelerator="cpu", devices=1, logger=False,
        enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        limit_val_batches=0, num_sanity_val_steps=0, default_root_dir=tmp_path,
    )
    trainer.fit(module, train_dataloaders=torch.utils.data.DataLoader(batch, batch_size=5))
    assert trainer.global_step == 2
    assert torch.isfinite(trainer.callback_metrics["training/loss"])
    # A loss for one head must leave every other head's dictionary untouched.
    module.zero_grad(set_to_none=True)
    sae(batch[:, 0])[2][:, 0].sum().backward()
    for parameter in (sae.encoder.W, sae.encoder.b, sae.decoder.W, sae.decoder.b):
        assert parameter.grad[0].abs().sum() > 0
        assert torch.count_nonzero(parameter.grad[1:]) == 0
