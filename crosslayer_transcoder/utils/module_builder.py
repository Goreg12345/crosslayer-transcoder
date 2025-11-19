import importlib
import torch.nn as nn


def build_module_from_config(config: dict) -> nn.Module:
    """Naive module builder that constructs a module from a yaml config."""
    class_path = config["class_path"]
    module_name = ".".join(class_path.split(".")[:-1])
    class_name = class_path.split(".")[-1]
    init_class = importlib.import_module(module_name)
    init_class = getattr(init_class, class_name)
    init_args = config.get("init_args", {})

    if not init_args:
        return init_class()

    for key, value in init_args.items():
        if isinstance(value, dict) and "class_path" in value:
            init_args[key] = build_module_from_config(value)

    init_class = init_class(**init_args)

    return init_class
