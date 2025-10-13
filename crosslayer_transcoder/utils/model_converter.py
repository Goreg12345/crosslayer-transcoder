from abc import ABC, abstractmethod
from typing import Union

import lightning as L

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
    def __init__(
        self,
    ):
        pass

    @abstractmethod
    def convert(self, model: CLTModule) -> CLTModule:
        pass
