from typing import Protocol, Union, runtime_checkable

import torch

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


@runtime_checkable
class ModelConverter(Protocol):
    def convert_and_save(
        self,
        model: CLTModule,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        pass
