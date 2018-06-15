import torch.nn as nn


class TorchModel(BaseModel):
    r""" Base class for all torch models
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.module = nn.Module()
