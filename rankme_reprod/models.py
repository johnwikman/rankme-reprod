import torch
import torch.nn as nn


class EncoderProjector(nn.Module):
    """
    Classifier with latent state, as described in the RankMe paper.
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
