
def test_load_model_from_lightning_checkpoint(clt_module):
    from crosslayer_transcoder.utils.checkpoints import load_model_from_lightning_checkpoint
    checkpoint_path = "crosslayer_transcoder/checkpoints/clt.ckpt"

    model = load_model_from_lightning_checkpoint(clt_module, checkpoint_path)

    print(model)
