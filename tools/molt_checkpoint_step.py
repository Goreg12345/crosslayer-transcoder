"""Print completed optimizer steps from a local Lightning checkpoint."""
import sys
import torch


def checkpoint_step(path):
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    step = checkpoint.get('global_step')
    if not isinstance(step, int) or step < 0:
        raise ValueError(f'{path}: missing or invalid global_step')
    return step


if __name__ == '__main__':
    print(checkpoint_step(sys.argv[1]))
