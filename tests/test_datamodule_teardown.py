from crosslayer_transcoder.data.datamodule import ActivationDataModule


class _TruthinessMustNotRun:
    def __bool__(self):
        raise AssertionError("teardown evaluated object truthiness")


class _Buffer(_TruthinessMustNotRun):
    def __init__(self):
        self.cleaned = False

    def cleanup(self):
        self.cleaned = True


class _Generator(_TruthinessMustNotRun):
    def is_alive(self):
        return False


def test_teardown_does_not_require_dataloader_or_buffer_len():
    datamodule = ActivationDataModule(buffer_size=1, batch_size=1)
    datamodule.data_loader = _TruthinessMustNotRun()
    datamodule.data_generator = _Generator()
    datamodule.shared_buffer = _Buffer()

    datamodule.teardown("fit")

    assert datamodule.shared_buffer.cleaned
