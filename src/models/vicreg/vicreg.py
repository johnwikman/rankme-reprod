# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
import torch.nn as nn
import torch.nn.functional as F

from ..pretrainer import ImagePretrainer

import logging

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class VICReg(ImagePretrainer):
    def __init__(
        self,
        sim_coeff=25.0,
        std_coeff=25.0,
        cov_coeff=1.0,
        *args,
        **kwargs
    ) -> None:
        """
        Parameters
        sim_coeff : float
          Similarity (Mean-Squared Error) loss weight.

        std_coeff : float
          Standard deviation loss weight.

        cov_coeff : float
          Covariance loss weight.
        """
        super().__init__(*args, **kwargs)
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

    def train_iter(self, images, compute_stats=False) -> None:
        """
        A single iteration of VICReg training.

        Computes the VICReg loss between batched output embeddings from an
        EncoderProjector network, where x and y are the output embedding of
        the same image with slightly different augmentations. The shape of
        both x and y should be [N, E] where N is the number of embeddings and
        E is the number of features (or encoding dimensionality).
        """
        # Slow (but safe) way
        #img_x, img_y = images
        #x = self.model(img_x.to(self.device))
        #y = self.model(img_y.to(self.device))

        # Fast way (but less obvious)
        images = torch.cat(images, dim=0)
        images = images.to(self.device)
        x, y = torch.split(
            self.model(images),
            [self.batch_size, self.batch_size]
        )

        repr_loss = F.mse_loss(x, y)

        # Convert x and y to have batch mean µ = 0
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_loss = self.covariance_loss(x, y)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss, {}

    def covariance_loss(self, x, y):
        """Returns the covariance loss, this should be overridden by VICReg-ctr and VICReg-exp."""
        batch_size, num_features = x.shape

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(num_features)

        return cov_loss


class VICRegExp(VICReg):
    def __init__(self, temperature=1.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def covariance_loss(self, x, y):
        """Different loss according to RankMe Paper."""
        batch_size, num_features = x.shape

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = torch.log(
            torch.exp(off_diagonal(cov_x) / self.temperature).sum() +
            torch.exp(off_diagonal(cov_y) / self.temperature).sum() +
            1e-7
        ).div(num_features)

        return cov_loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
