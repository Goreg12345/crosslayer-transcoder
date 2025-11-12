import nnsight
import torch

gpt2 = nnsight.LanguageModel(
    "openai-community/gpt2", device_map="cuda:0", dispatch=True
)

gpt2.requires_grad_(False)


while True:
    tokens = torch.randint(0, gpt2.config.vocab_size, (5, 1024))
    mean = torch.zeros(1, device="cuda:0")
    with gpt2.trace(tokens):
        features = torch.full(
            (tokens.shape[0], tokens.shape[1], gpt2.config.n_layer, gpt2.config.n_embd),
            float("nan"),
            device=tokens.device,
        )
        actvs = gpt2.transformer.h[0].mlp.output.save()
        b, l, d = actvs.shape

        vals, idxs = torch.topk(actvs, k=3)
    mean += actvs.mean()
