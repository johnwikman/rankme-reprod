import torch
import torch.nn as nn
import torchvision.models


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


def make_resnet18():
    # Original paper had 8192, 8192, 2048 as hidden dimensions, but we instead
    # use 2048, 2048, 1024 since that is more managable.
    encoder = torchvision.models.resnet18(num_classes=2048)

    projector_in_dim = encoder.fc.in_features

    projector = nn.Sequential(
        nn.Linear(projector_in_dim, 2048),
        nn.ReLU(),  # 2048
        nn.Linear(2048, 2048),
        nn.ReLU(),  # 2048
        nn.Linear(2048, 1024),
    )
    encoder.fc = nn.Identity()

    model = EncoderProjector(
        encoder=encoder,
        projector=projector,
    )

    return model


MODELS = {
    "resnet18": make_resnet18,
}
