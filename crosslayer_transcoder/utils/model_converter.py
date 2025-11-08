from abc import ABC, abstractmethod
from typing import Union

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


class ModelConverter(ABC):
    @abstractmethod
    def convert(self, model: CLTModule) -> CLTModule:
        pass
