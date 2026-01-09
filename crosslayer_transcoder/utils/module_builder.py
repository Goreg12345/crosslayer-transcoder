import importlib
import torch.nn as nn


def build_module_from_config(config: dict) -> nn.Module:
    """Naive module builder that constructs a module from a yaml config."""
    class_path = config["class_path"]
    module_name, class_name = class_path.rsplit(".", 1)
    init_class = importlib.import_module(module_name)
    init_class = getattr(init_class, class_name)
    init_args = config.get("init_args", {})

    cls = getattr(importlib.import_module(module_name), class_name)

    return cls.from_config(init_args)
