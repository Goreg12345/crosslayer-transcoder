"""Count non-padding training tokens after the exact chat template and truncation."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('RAYON_NUM_THREADS', '4')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'true')
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen3-4B')
    parser.add_argument('--seq-len', type=int, default=1024)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cache = Path.home() / '.cache/huggingface/datasets/HuggingFaceH4___ultrachat_200k/default/0.0.0/8049631c405ae6576f93f445c6b8166f76f5505a'
    paths = sorted(cache.glob('ultrachat_200k-train_sft-*.arrow'))
    if len(paths) != 3:
        raise RuntimeError(f'Expected all three cached train_sft shards, found {paths}')
    dataset = concatenate_datasets([Dataset.from_file(str(p)) for p in paths])
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    total = truncated = rows = 0
    started = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for start in range(0, len(dataset), 256):
        conversations = dataset[start:start+256]['messages']
        batches = tokenizer.apply_chat_template(conversations, tokenize=True, add_generation_prompt=False)
        if start == 0:
            for conversation, ids in zip(conversations[:8], batches[:8]):
                assert ids == tokenizer.apply_chat_template(conversation, tokenize=True, add_generation_prompt=False)
        total += sum(min(len(ids), args.seq_len) for ids in batches)
        truncated += sum(len(ids) > args.seq_len for ids in batches)
        rows += len(batches)
        if start % (256 * 20) == 0 or rows == len(dataset):
            result = dict(model=args.model, dataset='HuggingFaceH4/ultrachat_200k', split='train_sft',
                          dataset_fingerprint=dataset._fingerprint, max_sequence_length=args.seq_len,
                          add_generation_prompt=False, processed_rows=rows, dataset_rows=len(dataset),
                          non_padding_tokens=total, truncated_rows=truncated, complete=rows == len(dataset),
                          elapsed_seconds=round(time.time()-started, 1))
            temporary = args.output.with_suffix('.tmp')
            temporary.write_text(json.dumps(result, indent=2)+'\n')
            temporary.replace(args.output)
            print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
