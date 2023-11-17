#!/usr/bin/env python3

import torch
import torch.nn as nn

from torchvision.models import resnet18, resnet50


def main():
    N_CLASSES = 1000

    encoder = resnet18(num_classes=8192)
    reprfeats = encoder.fc.in_features
    encoder.fc = nn.Identity() # bypass the FC off

    projector = nn.Sequential(
        nn.Linear(reprfeats, 8192)
        # (8192)
        nn.Linear(8192, 8192),
        # (8192)
        nn.Linear(8192, 2048),
        # (2048)
        nn.Linear(2048, N_CLASSES),
    )

    input = "image from imagenet"
    representation = encoder(input)
    embedding = projector(representation)

    def rankme(z, epsilon=1e-6):
        sigmas = torch.linalg.svdvals(z)
        norm = sigmas.abs().sum(dim=-1)
        p = sigmas / norm.unsqueeze(-1) + epsilon

        return torch.exp(-(torch.log(p) * p).sum(dim=-1))





if __name__ == "__main__":
    main()
