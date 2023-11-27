"""
A wrapper class for SimCLR training.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ...utils.evaluate import topk_accuracy

import logging
import mlflow

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class SimCLRLoss(nn.Module):
    """
    Implements the Info NCE loss that is used in SimCLR. Use this as:

        criterion = SimCLRLoss()
        x, y = augment1(imgs), augment2(imgs)
        loss = criterion(x, y)
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.register_buffer("temperature", torch.tensor(temperature, dtype=torch.float32))
        self.register_buffer("criterion", torch.nn.CrossEntropyLoss())

    def forward(self, x, y):
        assert x.shape == y.shape
        assert len(x.shape) == 2
        batch_size, embedding_dim = x.shape
        device = self.temperature.device
        features = torch.cat([x, y], dim=0).to(device)

        labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # TODO: leaving debugging code in for now
        # assert similarity_matrix.shape == (n_views * batch_size, n_views * batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
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
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)

        logits = logits / self.temperature
        return self.criterion(logits, labels)



class SimCLR:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        fp16_precision,
        epochs,
        batch_size,
        learning_rate,
        temperature,
        n_views,
        device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.fp16_precision = fp16_precision
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.temperature = temperature
        self.n_views = n_views
        self.device = device

    # @torch.jit.script
    def _info_nce_loss(
        self,
        features: torch.Tensor,
        n_views: int,
        batch_size: int,
    ):
        labels = torch.cat([torch.arange(batch_size) for _ in range(n_views)], dim=0)
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

    def train(self, train_loader) -> None:
        """
        The main training loop for SimCLR.
        NOTE: mutates self.model, instead of returning it.
        """
        if self.fp16_precision:
            from torch.cuda.amp import GradScaler, autocast

            scaler = GradScaler(enabled=True)

        criterion = torch.nn.CrossEntropyLoss().to(self.device)

        n_iter = 0
        LOG.info(f"Start SimCLR training for {self.epochs} epochs.")
        LOG.info(f"(Using 16-bit floating point precision: {self.fp16_precision})")

        for epoch_counter in range(self.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)
                images = images.to(self.device)

                self.optimizer.zero_grad()

                if self.fp16_precision:
                    with autocast(enabled=True):
                        features = self.model(images)
                        logits, labels = self._info_nce_loss(
                            features,
                            n_views=self.n_views,
                            batch_size=self.batch_size,
                        )
                        loss = criterion(logits, labels)
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                else:
                    features = self.model(images)
                    logits, labels = self._info_nce_loss(
                        features,
                        n_views=self.n_views,
                        batch_size=self.batch_size,
                    )
                    loss = criterion(logits, labels)
                    loss.backward()
                    self.optimizer.step()

                    if n_iter % 100 == 0:  # args.log_every_n_steps == 0:
                        top1, top5 = topk_accuracy(logits, labels, topk=(1, 5))

                        mlflow.log_metric("loss", loss.item(), step=n_iter)
                        mlflow.log_metric("acc/top1", top1[0].item(), step=n_iter)
                        mlflow.log_metric("acc/top5", top5[0].item(), step=n_iter)
                        mlflow.log_metric(
                            "learning_rate",
                            self.scheduler.get_last_lr()[0],
                            step=n_iter,
                        )

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            LOG.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

    def save(self, model_path):
        """
        Save model to model_path and log to MLflow.
        First, move it to CPU for compliance.
        """
        self.model.to("cpu")
        torch.save(self.model, model_path)

        # Log the model to MLflow
        # "pretraining_model" used when loading the artifact downstream
        mlflow.pytorch.log_model(self.model, "pretraining_model")
