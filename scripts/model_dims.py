"""Print a HF model's (hidden_size, num_hidden_layers) for config wiring.

MoLT training needs the base model's residual width (`d_acts` / `activation_dim`)
and block count (`n_layers`). Rather than hardcode possibly-wrong numbers for a
given model, derive them at runtime from the model config:

    uv run python scripts/model_dims.py google/gemma-3-4b-it
    # -> hidden_size=2560 num_hidden_layers=34   (example)

Output formats:
    (default)      human-readable "hidden_size=.. num_hidden_layers=.."
    --shell        "<hidden_size> <num_hidden_layers>"  (read DIM L < <(...))
    --field FIELD  print a single value (hidden_size | num_hidden_layers)

Gemma 3 (and other multimodal) configs nest the text tower under
`text_config`; this script unwraps that automatically.
"""

from __future__ import annotations

import argparse
import sys

from transformers import AutoConfig


def get_dims(model_name: str) -> tuple[int, int]:
    cfg = AutoConfig.from_pretrained(model_name)
    # Multimodal wrappers (e.g. Gemma 3) keep the text tower under text_config.
    text_cfg = getattr(cfg, "text_config", None) or cfg
    hidden = getattr(text_cfg, "hidden_size", None)
    layers = getattr(text_cfg, "num_hidden_layers", None)
    if hidden is None or layers is None:
        raise ValueError(
            f"Could not find hidden_size/num_hidden_layers on {model_name} "
            f"(got hidden_size={hidden}, num_hidden_layers={layers})"
        )
    return int(hidden), int(layers)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model_name")
    p.add_argument("--shell", action="store_true", help="print '<hidden> <layers>' for shell capture")
    p.add_argument("--field", choices=["hidden_size", "num_hidden_layers"], default=None)
    args = p.parse_args(argv)

    hidden, layers = get_dims(args.model_name)
    if args.field == "hidden_size":
        print(hidden)
    elif args.field == "num_hidden_layers":
        print(layers)
    elif args.shell:
        print(f"{hidden} {layers}")
    else:
        print(f"hidden_size={hidden} num_hidden_layers={layers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
