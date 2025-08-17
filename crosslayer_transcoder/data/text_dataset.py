import os
from typing import Callable

import torch
from torch.utils.data import IterableDataset


def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    num_workers = worker_info.num_workers  # number of workers
    pointer_start = int(worker_id / num_workers * len(dataset.hf_dataset))
    dataset.set_token_pointer(pointer_start)


class TextDataset(IterableDataset):
    def __init__(
        self,
        hf_dataset,
        to_tokens: Callable,
        batch_size,
        drop_last_batch=True,
        hf_text_accessor="text",
        seq_len=128,
    ):
        """
        Takes a huggingface dataset and returns batches of tokens and their attention masks (for padding)
        :param hf_dataset: huggingface dataset that contains the text
        :param to_tokens: function that converts text to tokens, e.g. the tokenizer function or HookedTransformer.to_tokens()
        :param batch_size: batch size
        :param drop_last_batch: if True, the last batch will be dropped if it's smaller than batch_size
        :param hf_text_accessor: str, key to access the text in the hf_dataset
        :param seq_len: int, sequence length per sample in the batch
        returns batches of shape (batch_size, seq_len), filled with tokens and their respective attention masks for padding
        """
        self.hf_dataset = hf_dataset
        self.to_tokens = to_tokens
        self.token_pointer = 0
        self.drop_last_batch = drop_last_batch
        self.hf_text_accessor = hf_text_accessor
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.batch_pointer = 0
        self.tokens = []

    def set_token_pointer(self, pointer_start):  # for multi-process dataloader
        # if e.g. 2 workers, the first worker will start at 0, the second at 1/2 of the dataset
        self.token_pointer = pointer_start

    def __iter__(self):
        return self

    def __next__(self):
        batch = torch.zeros((self.batch_size, self.seq_len), dtype=torch.long)
        mask = torch.zeros((self.batch_size, self.seq_len), dtype=torch.bool)

        # if dataset is exhausted, stop
        if self.token_pointer == len(self.hf_dataset):
            raise StopIteration

        # get a new sample and add it to the batch
        for b_idx in range(self.batch_size):
            tokens = self.to_tokens(
                self.hf_dataset[self.token_pointer][self.hf_text_accessor],
            )["input_ids"]
            batch[b_idx, : min(len(tokens), self.seq_len)] = torch.tensor(
                tokens[: self.seq_len], dtype=torch.long
            )
            mask[b_idx, : min(len(tokens), self.seq_len)] = True
            self.token_pointer += 1
            if self.token_pointer == len(self.hf_dataset):
                return batch, mask  # remaining sequences are padded with 0s

        return batch, mask
