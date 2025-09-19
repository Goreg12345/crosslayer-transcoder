import psutil
import torch
from rich.console import Console

from crosslayer_transcoder.modal.app import CHECKPOINTS_PATH, app, volumes

console = Console()


def get_memory_usage():
    """Get current memory usage statistics"""
    process = psutil.Process()
    memory_info = process.memory_info()

    stats = {
        "cpu_memory_mb": memory_info.rss / 1024 / 1024,
        "cpu_memory_percent": process.memory_percent(),
    }

    if torch.cuda.is_available():
        stats.update(
            {
                "gpu_memory_allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
                "gpu_memory_reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
                "gpu_memory_max_allocated_mb": torch.cuda.max_memory_allocated()
                / 1024
                / 1024,
            }
        )

    return stats


def log_memory_usage(stage: str):
    """Log memory usage with a descriptive stage label"""
    stats = get_memory_usage()
    console.print(f"[bold blue]{stage}[/bold blue]")
    console.print(
        f"  CPU Memory: {stats['cpu_memory_mb']:.1f} MB ({stats['cpu_memory_percent']:.1f}%)"
    )

    if torch.cuda.is_available():
        console.print(
            f"  GPU Memory Allocated: {stats['gpu_memory_allocated_mb']:.1f} MB"
        )
        console.print(
            f"  GPU Memory Reserved: {stats['gpu_memory_reserved_mb']:.1f} MB"
        )
        console.print(
            f"  GPU Memory Max: {stats['gpu_memory_max_allocated_mb']:.1f} MB"
        )


@app.function(gpu="a10g", volumes=volumes)
def load_ckpt_lightning(ckpt_path):
    from crosslayer_transcoder.model.clt_lightning import (
        JumpReLUCrossLayerTranscoderModule,
    )
    from crosslayer_transcoder.model.clt import CrossLayerTranscoder
    from crosslayer_transcoder.model.clt import Encoder
    from crosslayer_transcoder.model.clt import CrosslayerDecoder
    from crosslayer_transcoder.model.standardize import DimensionwiseInputStandardizer
    from crosslayer_transcoder.model.standardize import DimensionwiseOutputStandardizer
    from crosslayer_transcoder.model.jumprelu import JumpReLU
    from crosslayer_transcoder.metrics.replacement_model_accuracy import (
        ReplacementModelAccuracy,
    )
    from crosslayer_transcoder.metrics.dead_features import DeadFeatures

    console.print("[bold yellow]Loading model from checkpoint...[/bold yellow]")

    log_memory_usage("Initial memory usage")

    encoder = Encoder(
        d_acts=768,
        d_features=10_000,
        n_layers=12,
    )
    decoder = CrosslayerDecoder(
        d_acts=768,
        d_features=10_000,
        n_layers=12,
    )

    input_standardizer = DimensionwiseInputStandardizer(
        n_layers=12,
        activation_dim=768,
    )
    output_standardizer = DimensionwiseOutputStandardizer(
        n_layers=12,
        activation_dim=768,
    )
    nonlinearity = JumpReLU(theta=0.03, bandwidth=0.01, n_layers=12, d_features=10_000)

    clt = CrossLayerTranscoder(
        encoder=encoder,
        decoder=decoder,
        input_standardizer=input_standardizer,
        output_standardizer=output_standardizer,
        nonlinearity=nonlinearity,
    )

    log_memory_usage("After model creation")

    model = JumpReLUCrossLayerTranscoderModule.load_from_checkpoint(
        ckpt_path,
        model=clt,
        replacement_model=None,
        dead_features=None,
    )
    model.eval()

    log_memory_usage("After checkpoint loading")

    console.print("[bold green]✓ Model loaded successfully![/bold green]")
    return


@app.local_entrypoint()
def main():
    experiment = "34f20c70"
    console.print(
        f"[bold cyan]Starting checkpoint loading test for experiment:[/bold cyan] {experiment}"
    )
    load_ckpt_lightning.remote(f"{CHECKPOINTS_PATH}/{experiment}/last.ckpt")
