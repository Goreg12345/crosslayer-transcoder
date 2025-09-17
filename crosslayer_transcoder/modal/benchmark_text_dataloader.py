from crosslayer_transcoder.modal.app import app, volumes
from tests.benchmark_text_dataloader import benchmark_textdataset


@app.function(
    gpu="a10g",
    timeout=60 * 60 * 24,
    volumes=volumes,
)
def benchmark_text_dataloader():
    # TODO: This doesn't work quite yet
    benchmark_textdataset(
        dataset_name="Skylion006/openwebtext",
        model_name="openai-community/gpt2",
        batch_size=32,
        seq_len=1024,
        num_workers=0,
        max_batches=10,
        max_duration=60.0,
    )


@app.local_entrypoint()
def main():
    benchmark_text_dataloader.remote()
