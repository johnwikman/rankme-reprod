# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F



# This is cosine annealing LR with 10 epochs of warmup
# This exists as CosineAnnealingLR(T_max=len(loader), eta_min=base_lr*0.001)
# The base_lr is implicit from the underlying optimizer.
#def adjust_learning_rate(args, optimizer, loader, step):
#    max_steps = args.epochs * len(loader)
#    warmup_steps = 10 * len(loader)
#    base_lr = args.base_lr * args.batch_size / 256
#    if step < warmup_steps:
#        lr = base_lr * step / warmup_steps
#    else:
#        step -= warmup_steps
#        max_steps -= warmup_steps
#        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
#        end_lr = base_lr * 0.001
#        lr = base_lr * q + end_lr * (1 - q)
#    for param_group in optimizer.param_groups:
#        param_group["lr"] = lr
#    return lr


class VICRegLoss(nn.Module):
    """
    Example usage:

        model = EncoderProjector(encoder=my_resnet, projector=my_proj)
        criterion = VICRegLoss()
        for epoch in range(epochs):
            for (x, y) in dataloader:
                optimizer.zero_grad()
                loss = criterion(model(x), model(y))
                loss.backward()
                optimizer.step()

    Note that, in the reference architecture of VICReg, they perform batch
    normalization before the ReLU in the MLP:

        modules = []
        for i in range(len(layers)):
            modules += [
                nn.Linear(layers[i], layers[i+1]),
                nn.BatchNorm1d(layers[i+1]),
                nn.ReLU(true),
            ]
        modules.append(nn.Linear(layers[-2], layers[-1]))
        return nn.Sequential(*modules)
    """
    def __init__(self,
                 sim_coeff=25.0,
                 std_coeff=25.0,
                 cov_coeff=1.0):
        """
        Parameters
        sim_coeff : float
          Similarity (Mean-Squared Error) loss weight.

        std_coeff : float
          Standard deviation loss weight.

        cov_coeff : float
          Covariance loss weight.
        """
        super().__init__()
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0

    def forward(self, x, y):
        """
        Computes the VICReg loss between batched output embeddings from an
        EncoderProjector network, where x and y are the output embedding of
        the same image with slightly different augmentations. The shape of
        both x and y should be [N, E] where N is the number of embeddings and
        E is the number of features (or encoding dimensionality).
        """
        assert len(x.shape) == 2
        assert x.shape == y.shape
        batch_size, num_features = x.shape

        repr_loss = F.mse_loss(x, y)

        # Convert x and y to have batch mean µ = 0
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_y = (y.T @ y) / (batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(num_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
