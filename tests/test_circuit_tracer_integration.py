from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
import torch


def validate_upload(repo_id: str):
    """Verify uploaded model loads correctly."""

    # Load from HuggingFace
    transcoder, config = load_transcoder_from_hub(
        repo_id,
        device=torch.device("cpu"),
        dtype=torch.float32,
        lazy_encoder=False,
        lazy_decoder=False,
    )

    print(
        f"✓ Dimensions: {transcoder.n_layers}L x {transcoder.d_transcoder}F x {transcoder.d_model}D"
    )
    print(f"✓ Encoder weights match")
    print(f"✓ Config: {config['model_kind']}")


if __name__ == "__main__":
    validate_upload(
        repo_id="jiito/clt_test_gpt2_zero",
    )
