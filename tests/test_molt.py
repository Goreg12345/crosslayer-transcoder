import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from crosslayer_transcoder.model.jumprelu import JumpReLU
from crosslayer_transcoder.model.molt import Molt
from crosslayer_transcoder.model.standardize import (
    DimensionwiseInputStandardizer,
    DimensionwiseOutputStandardizer,
)
from crosslayer_transcoder.data.activation_sources import ActivationComputer
from crosslayer_transcoder.data.text_dataset import TextDataset


D_ACTS = 7
N_LAYERS = 3
N = 2
RANKS = [4, 2, 1]
BATCH_SIZE = 5


def build_molt() -> Molt:
    n_features = N * sum(2**index for index in range(len(RANKS)))
    model = Molt(
        d_acts=D_ACTS,
        N=N,
        ranks=RANKS,
        nonlinearity=JumpReLU(
            theta=0.03,
            bandwidth=1.0,
            n_layers=1,
            d_features=n_features,
        ),
        input_standardizer=DimensionwiseInputStandardizer(
            n_layers=N_LAYERS, activation_dim=D_ACTS
        ),
        output_standardizer=DimensionwiseOutputStandardizer(
            n_layers=N_LAYERS, activation_dim=D_ACTS
        ),
    )
    initialization_batch = torch.randn(BATCH_SIZE, 2, N_LAYERS, D_ACTS)
    model.initialize_standardizers(initialization_batch)
    return model


def legacy_forward(model: Molt, acts: torch.Tensor, layer: int):
    """Independent expression of the pre-refactor MOLT forward equations."""
    standardized = (acts - model.input_standardizer.mean[layer]) / model.input_standardizer.std[layer]
    pre_activations = standardized @ model.e.weight.T + model.e.bias
    gate = model.nonlinearity(pre_activations)

    reconstructions = []
    for U, V in zip(model.Us, model.Vs):
        latents = torch.einsum("bd,ndr->bnr", standardized, V)
        reconstructions.append(torch.einsum("bnr,nrd->bnd", latents, U))
    reconstruction_norm = (gate.unsqueeze(-1) * torch.cat(reconstructions, dim=1)).sum(dim=1)
    reconstruction = (
        reconstruction_norm * model.output_standardizer.std[layer]
        + model.output_standardizer.mean[layer]
    )
    return gate, reconstruction_norm, reconstruction


def legacy_transform_norm(model: Molt):
    transforms = [torch.einsum("nro,ndr->ndo", U, V) for U, V in zip(model.Us, model.Vs)]
    return torch.cat([transform.norm(dim=(1, 2)) for transform in transforms])


def test_molt_parameter_shapes_and_derived_sizes():
    model = build_molt()

    assert model.n_features == N * (1 + 2 + 4)
    assert model.d_latents == N * (RANKS[0] + 2 * RANKS[1] + 4 * RANKS[2])
    assert [tuple(parameter.shape) for parameter in model.Us] == [
        (N, RANKS[0], D_ACTS),
        (2 * N, RANKS[1], D_ACTS),
        (4 * N, RANKS[2], D_ACTS),
    ]


def test_molt_forward_and_transform_norm_match_legacy_equations():
    torch.manual_seed(0)
    model = build_molt()
    acts = torch.randn(BATCH_SIZE, D_ACTS)

    actual = model(acts, layer=1)
    expected = legacy_forward(model, acts, layer=1)

    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(model.transform_norm(), legacy_transform_norm(model))


def test_molt_forward_with_pre_activations_preserves_public_forward():
    model = build_molt()
    acts = torch.randn(BATCH_SIZE, D_ACTS)

    pre_activations, *outputs = model.forward_with_pre_activations(acts, layer=1)

    assert pre_activations.shape == (BATCH_SIZE, model.n_features)
    for actual, expected in zip(outputs, model(acts, layer=1)):
        torch.testing.assert_close(actual, expected)


def test_molt_gradients_match_legacy_equations():
    torch.manual_seed(1)
    actual_model = build_molt()
    expected_model = build_molt()
    expected_model.load_state_dict(actual_model.state_dict())
    acts = torch.randn(BATCH_SIZE, D_ACTS)

    actual_outputs = actual_model(acts, layer=2)
    expected_outputs = legacy_forward(expected_model, acts, layer=2)
    (actual_outputs[1].square().mean() + actual_model.transform_norm().sum()).backward()
    (expected_outputs[1].square().mean() + legacy_transform_norm(expected_model).sum()).backward()

    for (actual_name, actual_parameter), (expected_name, expected_parameter) in zip(
        actual_model.named_parameters(), expected_model.named_parameters()
    ):
        assert actual_name == expected_name
        torch.testing.assert_close(
            actual_parameter.grad, expected_parameter.grad, rtol=1e-5, atol=1e-6
        )


def test_molt_save_and_load_round_trip_preserves_outputs():
    torch.manual_seed(2)
    model = build_molt()
    acts = torch.randn(BATCH_SIZE, D_ACTS)
    expected = model(acts, layer=0)

    with tempfile.TemporaryDirectory() as tmpdir:
        model.save_pretrained(Path(tmpdir))
        loaded = Molt.from_pretrained(tmpdir)

    # Initialization state is runtime metadata in the existing standardizers.
    loaded.input_standardizer.is_initialized = True
    loaded.output_standardizer.is_initialized = True
    actual = loaded(acts, layer=0)

    assert loaded.N == model.N
    assert loaded.ranks == model.ranks
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor)


def test_molt_optimizer_step_is_finite():
    torch.manual_seed(3)
    model = build_molt()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    acts = torch.randn(BATCH_SIZE, D_ACTS)
    target = torch.randn(BATCH_SIZE, D_ACTS)

    gate, reconstruction_norm, _ = model(acts, layer=1)
    mse = (reconstruction_norm - model.output_standardizer.standardize(target, 1)).square().mean()
    sparsity = torch.tanh(model.transform_norm() * gate * 100.0).sum(dim=-1).mean() * 1.5e-4
    loss = mse + sparsity
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert torch.isfinite(parameter).all()


def test_split_gpu_gemma_config_and_memory_budget():
    config_path = Path("config/molt-gemma3-4b-it-5090-extractor-rtx6000.yaml")
    config = yaml.safe_load(config_path.read_text())
    data = config["data"]["init_args"]
    replacement = config["model"]["init_args"]["replacement_model"]

    assert config["trainer"]["devices"] == [0]
    assert data["device_map"] == "cuda:1"
    assert replacement is None
    assert data["model_name"] == "google/gemma-3-4b-it"
    assert data["model_arch"] == "gemma3"
    assert data["dataset_name"] == "HuggingFaceH4/ultrachat_200k"
    assert data["dataset_split"] == "train_sft"
    assert data["dataset_text_field"] == "messages"
    assert data["n_layers"] == 23
    assert data["batch_size"] == 1000
    assert data["generation_batch_size"] == 12
    assert data["dtype"] == "bfloat16"
    assert data["model_dtype"] == "bfloat16"
    assert config["trainer"]["precision"] == "bf16-mixed"
    assert config["trainer"]["accumulate_grad_batches"] == 1
    molt = config["model"]["init_args"]["model"]["init_args"]
    assert molt["N"] == 80
    assert molt["ranks"] == [512, 256, 128, 64, 32]
    assert molt["nonlinearity"]["init_args"]["d_features"] == 2480
    assert config["model"]["init_args"]["layer"] == 22

    element_size = torch.empty((), dtype=getattr(torch, data["dtype"])).element_size()
    buffer_bytes = (
        data["buffer_size"]
        * data["n_in_out"]
        * data["n_layers"]
        * data["activation_dim"]
        * element_size
    )
    assert 80_000_000_000 <= buffer_bytes < 100_000_000_000


def test_activation_computer_rejects_unknown_architecture():
    with pytest.raises(ValueError, match="Unsupported model_arch"):
        ActivationComputer(n_layers=1, model_arch="unknown")


def test_gemma_activation_locations_are_configurable():
    class Handle:
        def __init__(self, name):
            self.input = f"{name}.input"
            self.output = f"{name}.output"

    class Layer:
        pre_feedforward_layernorm = Handle("pre_norm")
        mlp = Handle("mlp")
        post_feedforward_layernorm = Handle("post_norm")

    class Model:
        class model:
            layers = [Layer()]

    computer = ActivationComputer(
        n_layers=1,
        model_arch="gemma3",
        input_location="post_norm",
        output_location="raw",
    )
    assert computer._layer_handles(Model(), 0) == ("pre_norm.output", "mlp.output")


def test_qwen_activation_locations():
    class Handle:
        def __init__(self, name):
            self.input = f"{name}.input"
            self.output = f"{name}.output"

    class Layer:
        post_attention_layernorm = Handle("pre_mlp_norm")
        mlp = Handle("mlp")

    class Model:
        class model:
            layers = [Layer()]

    computer = ActivationComputer(n_layers=1, model_arch="qwen3", output_location="raw")
    assert computer._layer_handles(Model(), 0) == ("pre_mlp_norm.input", "mlp.output")


def test_chat_messages_are_rendered_with_tokenizer_chat_template():
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi"},
    ]

    class ChatDataset:
        def __len__(self):
            return 1

        def __getitem__(self, index):
            return {"messages": messages}

    class ChatTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, conversation, **kwargs):
            self.calls.append((conversation, kwargs))
            return [11, 12, 13]

    tokenizer = ChatTokenizer()
    dataset = TextDataset(
        ChatDataset(),
        tokenizer,
        batch_size=1,
        hf_text_accessor="messages",
        seq_len=5,
    )
    tokens, mask = next(dataset)

    assert tokenizer.calls == [
        (messages, {"tokenize": True, "add_generation_prompt": False})
    ]
    torch.testing.assert_close(tokens, torch.tensor([[11, 12, 13, 0, 0]]))
    torch.testing.assert_close(mask, torch.tensor([[True, True, True, False, False]]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_molt_cuda_forward_backward(dtype):
    torch.manual_seed(4)
    model = build_molt().cuda()
    acts = torch.randn(BATCH_SIZE, D_ACTS, device="cuda")

    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        gate, reconstruction_norm, _ = model(acts, layer=1)
        loss = reconstruction_norm.square().mean() + gate.mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
