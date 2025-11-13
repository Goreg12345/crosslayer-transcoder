import datasets
import torch
from torch.utils.data import DataLoader

import crosslayer_transcoder.data.text_dataset as text_dataset


def get_webtext_dataloader(model, dataset_name="Skylion007/openwebtext", batch_size=40):
    dataset = datasets.load_dataset(dataset_name, split="train", trust_remote_code=True)
    token_dataset = text_dataset.TextDataset(
        dataset,
        model.tokenizer,
        batch_size,
        drop_last_batch=False,
        seq_len=1023,
    )
    text_dataset_loader = iter(
        DataLoader(
            token_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=5,
            prefetch_factor=5,
            worker_init_fn=text_dataset.worker_init_fn,
        )
    )
    return text_dataset_loader


def plot_actvs(actvs):
    if type(actvs) == list:
        # actvs: list(n_layers), every entry tensor batch_size x seq_len x d_acts
        actvs = torch.stack(
            [actv[0, 50, 300:500] for actv in actvs]
        )  # n_layers x d_acts
    else:  # batch_size x n_layers x d_acts
        actvs = actvs[0, :, 300:500]  # n_layers x d_acts
    actvs = actvs.cpu().numpy()

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Create a 3x4 grid of subplots
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()

    # Plot bar plots for each layer's activations
    for layer in range(actvs.shape[0]):
        actv = actvs[layer]
        sns.barplot(x=range(len(actv)), y=actv, ax=axes[layer])
        axes[layer].set_title(f"Layer {layer}")
        axes[layer].set_xlabel("Activation Index")
        axes[layer].set_ylabel("Activation Value")
        # remove xticks
        axes[layer].set_xticks([])

    # Adjust layout to prevent overlap
    plt.tight_layout()

