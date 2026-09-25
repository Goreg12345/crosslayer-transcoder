from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
import nnsight
from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from crosslayer_transcoder.data.activation_sources import ActivationComputer
from crosslayer_transcoder.data.datamodule import ActivationDataModule
from crosslayer_transcoder.model.clt_lightning import MoltModule
from crosslayer_transcoder.model.molt import Molt
from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.standardize import DimensionwiseInputStandardizer, DimensionwiseOutputStandardizer


def test_real_qwen_selected_extraction_matches_full_layer_slice():
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                        num_hidden_layers=3, num_attention_heads=4,
                        num_key_value_heads=2, head_dim=4)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({'[PAD]': 0, '[UNK]': 1}, unk_token='[UNK]')),
        pad_token='[PAD]', unk_token='[UNK]')
    model = nnsight.LanguageModel(Qwen3ForCausalLM(config).eval(), tokenizer=tokenizer)
    tokens = torch.tensor([[1, 2, 3], [4, 5, 0]])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    full = ActivationComputer(3, model_arch='qwen3').get_next_batch(model, tokens, mask)
    selected = ActivationComputer(1, model_arch='qwen3', layer=2).get_next_batch(model, tokens, mask)
    assert selected.shape == (5, 2, 1, 16)
    torch.testing.assert_close(selected, full[:, :, 2:3, :])


def test_selected_training_matches_loss_gradients_and_checkpoint_statistics():
    torch.manual_seed(42)
    model = Molt(d_acts=4, N=2, ranks=[2, 1],
                 nonlinearity=JumpReLU(theta=0.03, bandwidth=1.0, n_layers=1, d_features=6),
                 input_standardizer=DimensionwiseInputStandardizer(n_layers=3, activation_dim=4),
                 output_standardizer=DimensionwiseOutputStandardizer(n_layers=3, activation_dim=4))
    full = MoltModule(model=model, layer=2, c_sparsity=100)
    selected = MoltModule(model=deepcopy(model), layer=2, c_sparsity=100)
    for module in (full, selected):
        module._trainer = SimpleNamespace(max_steps=100, global_step=50,
                                           datamodule=None)
        module.log = lambda *a, **kw: None
    batch = torch.randn(8, 2, 3, 4)
    full_loss = full.training_step(batch, 0)
    selected_loss = selected.training_step(batch[:, :, 2:3, :], 0)
    torch.testing.assert_close(selected_loss, full_loss)
    full_loss.backward()
    selected_loss.backward()
    for p, q in zip(full.parameters(), selected.parameters()):
        torch.testing.assert_close(p.grad, q.grad)
    for name in ('input_standardizer', 'output_standardizer'):
        for stat in ('mean', 'std'):
            key = f'model.{name}.{stat}'
            assert selected.state_dict()[key].shape == (3, 4)
            torch.testing.assert_close(selected.state_dict()[key][2], full.state_dict()[key][2])
    # Forward still takes the actual LM layer index, as existing steering does.
    for actual, expected in zip(selected(batch[:, 0, 2], 2), full(batch[:, 0, 2], 2)):
        torch.testing.assert_close(actual, expected)

    # Resuming on a different first batch must retain checkpoint normalization.
    statistics = {k: v.clone() for k, v in selected.state_dict().items() if 'standardizer' in k}
    selected.on_load_checkpoint({'state_dict': selected.state_dict()})
    selected.training_step(batch[:, :, 2:3, :] + 100, 0)
    for key, value in statistics.items():
        torch.testing.assert_close(selected.state_dict()[key], value)


def test_single_layer_buffer_memory():
    data = ActivationDataModule(n_layers=1, activation_layer=35,
                                activation_dim=2560, buffer_size=8_800_000, dtype='bfloat16')
    expected_bytes = 8_800_000 * 2 * 2560 * 2 + 8_800_000
    assert data.get_memory_estimate_gb() == expected_bytes / 1024**3
    assert 90_000_000_000 < expected_bytes < 91_000_000_000
