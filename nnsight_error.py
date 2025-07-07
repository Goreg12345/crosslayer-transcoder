import nnsight
import torch

gpt2 = nnsight.LanguageModel(
    "openai-community/gpt2", device_map="cuda:0", dispatch=True
)

gpt2.requires_grad_(False)

with gpt2.trace("test"):
    actvs = gpt2.transformer.h[0].mlp.output
    shape = actvs.shape
    batch_size = shape[0]
    n_layers = shape[1]

    topk_result = torch.topk(actvs, k=3)
    vals = topk_result[0]
    idxs = topk_result[1]
