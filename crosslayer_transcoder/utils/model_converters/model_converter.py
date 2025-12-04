from typing import Any, Protocol, Union

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


class ModelConverter(Protocol):
    def convert_and_save(
        # NOTE: we type the model arg as any to avoid an issue with jsonargparse
        self, model: Any, dtype: torch.dtype = torch.bfloat16
    ) -> None:
        pass
