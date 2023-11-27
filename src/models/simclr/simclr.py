"""
A wrapper class for SimCLR training.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..pretrainer import ImagePretrainer
from ...utils.evaluate import topk_accuracy

import logging
import mlflow

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class SimCLR(ImagePretrainer):
    def __init__(
        self,
        temperature,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    # @torch.jit.script
    def _info_nce_loss(
        self,
        features: torch.Tensor,
        batch_size: int,
    ):
        labels = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # TODO: leaving debugging code in for now
        # assert similarity_matrix.shape == (n_views * batch_size, n_views * batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(
            similarity_matrix.shape[0], -1
        )
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.to(torch.bool)].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.to(torch.bool)].view(
            similarity_matrix.shape[0], -1
        )

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.temperature
        return logits, labels

    def train_iter(self, images, compute_stats=False) -> None:
        """
        A single iteration of training.
        """
        images = torch.cat(images, dim=0)
        images = images.to(self.device)

        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        stats = {}

        features = self.model(images)
        logits, labels = self._info_nce_loss(
            features,
            batch_size=self.batch_size,
        )
        loss = criterion(logits, labels)

        with torch.no_grad():
            if compute_stats:
                top1, top5 = topk_accuracy(logits, labels, topk=(1, 5))
                stats["acc/top1"] = top1[0].item()
                stats["acc/top5"] = top5[0].item()

        return loss, stats
