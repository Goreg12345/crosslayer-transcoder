import importlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Self, TypedDict, Union

import yaml
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from torch import nn


class ConfigDict(TypedDict):
    class_path: str
    init_args: Dict[str, Any]


class SerializableModule(nn.Module, ABC):
    """Base class for modules that can serialize to/from config and save/load to disk."""

    @abstractmethod
    def to_config(self) -> ConfigDict:
        """Serialize module configuration to a dict."""
        raise NotImplementedError

    @classmethod
    def from_config(cls, config: ConfigDict) -> Self:
        """Construct module from a config dict. Weights are not loaded."""
        init_args = config.get("init_args", {})

        resolved_args = {}

        for key, value in init_args.items():
            if isinstance(value, dict) and "class_path" in value:
                target_module_name, target_class_name = value["class_path"].rsplit(
                    ".", 1
                )
                target_cls = getattr(
                    importlib.import_module(target_module_name), target_class_name
                )
                resolved_args[key] = target_cls.from_config(value)
            else:
                resolved_args[key] = value

        return cls(**resolved_args)

    def save_pretrained(
        self,
        directory: Path,
    ) -> None:
        """Save config and weights to directory."""
        directory.mkdir(parents=True, exist_ok=True)

        with open(directory / "config.yaml", "w") as f:
            yaml.dump({"model": self.to_config()}, f)

        save_file(self.state_dict(), directory / "checkpoint.safetensors")

    @classmethod
    def from_pretrained(cls, directory: Union[Path, str]) -> Self:
        """Load model from local directory or HuggingFace Hub repository."""
        path = Path(directory)

        if not path.exists():
            local_path = snapshot_download(
                repo_id=str(directory),
                allow_patterns=["config.yaml", "checkpoint.safetensors"],
            )
            path = Path(local_path)

        with open(path / "config.yaml") as f:
            full_config = yaml.safe_load(f)

        model_config = full_config.get("model")
        if model_config is None:
            raise ValueError("Model config not found in config.yaml", full_config)

        if "class_path" in model_config:
            target_module_name, target_class_name = model_config["class_path"].rsplit(
                ".", 1
            )
            target_cls = getattr(
                importlib.import_module(target_module_name), target_class_name
            )
        else:
            target_cls = cls

        model = target_cls.from_config(model_config)
        model.load_state_dict(load_file(path / "checkpoint.safetensors"))

        model._is_folded = model_config.get("is_folded", False)

        return model
