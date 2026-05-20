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
        chat_template=False,
        add_generation_prompt=False,
    ):
        """
        Takes a huggingface dataset and returns batches of tokens and their attention masks (for padding)
        :param hf_dataset: huggingface dataset that contains the text
        :param to_tokens: function that converts text to tokens, e.g. the tokenizer function or HookedTransformer.to_tokens().
            When `chat_template=True` this must be a HuggingFace tokenizer (so `apply_chat_template` is available).
        :param batch_size: batch size
        :param drop_last_batch: if True, the last batch will be dropped if it's smaller than batch_size
        :param hf_text_accessor: str, key to access the text in the hf_dataset. With `chat_template=True` this
            column holds the chat messages (a list of {"role", "content"} dicts), e.g. "messages" for tulu-style data.
        :param seq_len: int, sequence length per sample in the batch
        :param chat_template: if True, render each sample with the tokenizer's chat template instead of tokenizing
            raw text. The leading BOS is stripped because the generation loop re-prepends one.
        :param add_generation_prompt: passed through to `apply_chat_template`. Leave False to tokenize the full
            conversation (user + assistant turns) for activation collection; set True to stop after the user turn.
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
        self.chat_template = chat_template
        self.add_generation_prompt = add_generation_prompt

    def set_token_pointer(self, pointer_start):  # for multi-process dataloader
        # if e.g. 2 workers, the first worker will start at 0, the second at 1/2 of the dataset
        self.token_pointer = pointer_start

    def __iter__(self):
        return self

    def _tokenize_sample(self, sample):
        """Return a flat list of token ids for one dataset row."""
        value = sample[self.hf_text_accessor]
        if not self.chat_template:
            return self.to_tokens(value)["input_ids"]

        # Chat path: `value` is a list of {"role", "content"} messages.
        ids = self.to_tokens.apply_chat_template(
            value,
            tokenize=True,
            add_generation_prompt=self.add_generation_prompt,
        )
        # The generation loop re-prepends BOS at position 0, so drop any
        # leading BOS the template added to avoid a double-BOS.
        bos_id = getattr(self.to_tokens, "bos_token_id", None)
        if bos_id is not None and len(ids) > 0 and ids[0] == bos_id:
            ids = ids[1:]
        return ids

    def __next__(self):
        batch = torch.zeros((self.batch_size, self.seq_len), dtype=torch.long)
        mask = torch.zeros((self.batch_size, self.seq_len), dtype=torch.bool)

        # if dataset is exhausted, stop
        if self.token_pointer == len(self.hf_dataset):
            raise StopIteration

        # get a new sample and add it to the batch
        for b_idx in range(self.batch_size):
            tokens = self._tokenize_sample(self.hf_dataset[self.token_pointer])
            batch[b_idx, : min(len(tokens), self.seq_len)] = torch.tensor(
                tokens[: self.seq_len], dtype=torch.long
            )
            mask[b_idx, : min(len(tokens), self.seq_len)] = True
            self.token_pointer += 1
            if self.token_pointer == len(self.hf_dataset):
                return batch, mask  # remaining sequences are padded with 0s

        return batch, mask
