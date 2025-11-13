import torch
from huggingface_hub import hf_hub_download
from crosslayer_transcoder.utils.model_converters.model_converter import CLTModule

def load_model_from_lightning_checkpoint(model, checkpoint_path):
    # NOTE: we might have to adjust how to resolve the map_location
    checkpoint = torch.load(checkpoint_path, map_location='cuda:0')
    model.load_state_dict(checkpoint["state_dict"])
    return model

def load_clt_from_hub(model: CLTModule, repo_id: str) -> CLTModule:
    local_filename = "clt.ckpt"
    checkpoint_path = hf_hub_download(repo_id=repo_id, local_dir=".", filename=local_filename)
    model = load_model_from_lightning_checkpoint(model, checkpoint_path)
    return model