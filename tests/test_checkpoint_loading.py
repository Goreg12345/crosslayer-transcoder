
# def test_load_model_from_lightning_checkpoint(clt_module):
#     checkpoint_path = "crosslayer_transcoder/checkpoints/clt.ckpt"

#     model = load_model_from_lightning_checkpoint(clt_module, checkpoint_path)

#     print(model)

def test_load_clt_from_hub(clt_module):
    from crosslayer_transcoder.utils.checkpoints import load_clt_from_hub
    model = load_clt_from_hub(clt_module, "georglange/gpt2-clt-topk-16-f-10k")
    print(model)