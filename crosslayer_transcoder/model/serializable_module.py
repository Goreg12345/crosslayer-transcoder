from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Self

import yaml
from safetensors.torch import load_file, save_file
from torch import nn


class SerializableModule(nn.Module, ABC):
    """Base class for modules that can serialize to/from config and save/load to disk."""

    @abstractmethod
    def to_config(self) -> Dict[str, Any]:
        """Serialize module configuration to a dict."""
        ...

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Self:
        """Construct module from a config dict. Weights are not loaded."""
        from crosslayer_transcoder.utils.module_builder import build_module_from_config

        init_args = config.get("init_args", {})
        resolved_args = {}

        for key, value in init_args.items():
            if isinstance(value, dict) and "class_path" in value:
                resolved_args[key] = build_module_from_config(value)
            else:
                resolved_args[key] = value

        return cls(**resolved_args)

    def save_pretrained(self, directory: Path) -> None:
        """Save config and weights to directory."""
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / "config.yaml", "w") as f:
            yaml.dump({"model": self.to_config()}, f)

        save_file(self.state_dict(), directory / "checkpoint.safetensors")

    @classmethod
    def from_pretrained(cls, directory: Path) -> Self:
        """Load model from directory."""
        with open(directory / "config.yaml") as f:
            full_config = yaml.safe_load(f)

        model_config = full_config.get("model")
        if model_config is None:
            raise ValueError("Model config not found in config.yaml", full_config)

        model = cls.from_config(model_config)
        model.load_state_dict(load_file(directory / "checkpoint.safetensors"))

        model._is_folded = model_config.get("is_folded", False)

        return model
