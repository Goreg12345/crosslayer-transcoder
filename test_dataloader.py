import time

import torch
from tqdm import tqdm

from buffer import DiscBuffer

buffer = DiscBuffer("/var/local/glang/activations/clt-activations-10M.h5", "tensor")


def main(n_batches=10):
    loader = torch.utils.data.DataLoader(
        buffer,
        num_workers=0,
        # prefetch_factor=2,
        batch_size=1000,
        shuffle=True,
        # persistent_workers=True,
        # pin_memory=True,
    )
    start_time = time.time()
    loader_iter = iter(loader)
    for i in tqdm(range(n_batches)):
        batch = next(loader_iter)  # Get one batch and move on
        if batch is None:
            break
    samples_per_second = 1000 * n_batches / (time.time() - start_time)
    print(f"Samples per second: {samples_per_second}")


if __name__ == "__main__":
    main()
