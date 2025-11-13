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
    # TODO: single resp
    def convert_and_save(self, model: CLTModule) -> None:
        pass
