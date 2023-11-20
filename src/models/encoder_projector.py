import torch
import torch.nn as nn


class EncoderProjector(nn.Module):
    """
    Used as a wrapper around a models describred in the RankeMe paper.
    Enables a modular way to swap out the encoder and projector.
    """

    def __init__(self, encoder, projector):
        super().__init__()
        self.encoder = encoder
        self.projector = projector

    def forward(self, x):
        z = self.encoder(x)
        return self.projector(z)

    def encode(self, x):
        return self.encoder(x)

    def project(self, z):
        return self.projector(z)
