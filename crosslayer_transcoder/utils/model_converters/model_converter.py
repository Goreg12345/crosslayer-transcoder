from typing import Protocol, Union

from crosslayer_transcoder.model.clt_lightning import (
    CrossLayerTranscoderModule,
    JumpReLUCrossLayerTranscoderModule,
    TopKCrossLayerTranscoderModule,
)

CLTModule = Union[
    CrossLayerTranscoderModule,
    JumpReLUCrossLayerTranscoderModule,
    TopKCrossLayerTranscoderModule,
]


class ModelConverter(Protocol):
    def convert_and_save(self, model: CLTModule) -> None: ...
