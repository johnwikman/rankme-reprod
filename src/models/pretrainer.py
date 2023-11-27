import torch
import torch.nn as nn
from tqdm import tqdm

import mlflow
import logging

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class ImagePretrainer:
    """
    Base class for all training algorithms.
    """
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        fp16_precision,
        epochs,
        device,
        batch_size,
        **kwargs
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.fp16_precision = fp16_precision
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        if len(kwargs) > 0:
            LOG.warning(f"Unused arguments: {kwargs}")

    def train_iter(self, images):
        raise NotImplementedError("subclass me")

    def train(self, dataloader):
        """
        The main training loop for SimCLR.
        NOTE: mutates self.model, instead of returning it.
        """
        self.model.to(self.device)

        if self.fp16_precision:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler(enabled=True)

        LOG.info(f"Start {type(self).__name__} training for {self.epochs} epochs.")
        LOG.info(f"(Using 16-bit floating point precision: {self.fp16_precision})")

        n_iter = 0
        top1acc = None
        for epoch_counter in range(self.epochs):
            for images, _ in tqdm(dataloader):
                compute_stats = bool(n_iter % 100 == 0)
                n_iter += 1

                self.optimizer.zero_grad()

                if self.fp16_precision:
                    with torch.autocast("cuda"):
                        loss, stats = self.train_iter(images, compute_stats)
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                else:
                    loss, stats = self.train_iter(images, compute_stats)
                    loss.backward()
                    self.optimizer.step()

                if compute_stats:
                    step = n_iter - 1
                    mlflow.log_metric("loss", loss.item(), step=step)
                    mlflow.log_metric(
                        "learning_rate",
                        self.scheduler.get_last_lr()[0],
                        step=step,
                    )
                    for k, v in stats.items():
                        mlflow.log_metric(k, v, step=step)
                        if "top1" in k:
                            top1acc = v

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            epoch_stats = [
                f"Epoch: {epoch_counter}",
                f"Loss: {loss.item()}",
            ]
            if top1acc is not None:
                epoch_stats.append(f"Top1 accuracy: {top1acc}")
            LOG.debug("\n".join(epoch_stats))

        LOG.info("Train loop done")

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
