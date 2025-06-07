import datasets
from torch.utils.data import DataLoader
import text_dataset


def get_webtext_dataloader(model):
    dataset = datasets.load_dataset('Skylion007/openwebtext', split='train')
    token_dataset = text_dataset.TextDataset(
        dataset,
        model.tokenizer,
        40,
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