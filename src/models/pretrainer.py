import torch
import torch.nn as nn
from tqdm import tqdm

import mlflow
import logging

from src.utils.evaluate import rank_me, get_rank


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
        use_target_rank_loss=False,
        target_rank_loss_opt=None,
        target_rank_loss_logalpha=None,
        target_rank=None,
        target_rank_dataloader=None,
        **kwargs
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.fp16_precision = fp16_precision
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.eval_dataloader = None
        self.use_target_rank_loss = use_target_rank_loss
        self.target_rank_loss_opt = target_rank_loss_opt
        self.target_rank_loss_logalpha = target_rank_loss_logalpha
        self.target_rank = target_rank
        self.target_rank_dataloader = target_rank_dataloader
        if self.use_target_rank_loss:
            if self.target_rank_loss_opt is None:
                raise ValueError("Missing target rank optimizer")
            if self.target_rank_loss_logalpha is None:
                raise ValueError("Missing target rank log(alpha) parameter")
            if self.target_rank is None:
                raise ValueError("Missing target rank hyperparameter")
            if self.target_rank_dataloader is None:
                raise ValueError("Missing target rank data loader")
        if len(kwargs) > 0:
            LOG.warning(f"Unused arguments: {kwargs}")

    def train_iter(self, images):
        raise NotImplementedError("subclass me")

    def set_eval_dataloader(self, dataloader):
        LOG.info("Setting eval_dataloader, will evaluate on this after every epoch.")
        self.eval_dataloader = dataloader

    def train(self, dataloader):
        """
        The main training loop for SimCLR.
        NOTE: mutates self.model, instead of returning it.
        """
        self.model.to(self.device)

        if self.fp16_precision:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler(enabled=True)

        # Wrap this to enable backwards compatibility with previous instances
        # that did not have a target rank loss member variable.
        use_target_rank_loss = False
        if "use_target_rank_loss" in self.__dict__:
            use_target_rank_loss = self.use_target_rank_loss
            target_rank_iterator = iter(self.target_rank_dataloader)

        LOG.info(f"Start {type(self).__name__} training for {self.epochs} epochs.")
        LOG.info(f"(Using 16-bit floating point precision: {self.fp16_precision})")
        LOG.info(f"(Using target rank loss: {use_target_rank_loss})")

        n_iter = 0
        top1acc = None
        for epoch_counter in range(self.epochs):
            self.model.train()
            for images, _ in tqdm(dataloader):
                compute_stats = bool(n_iter % 100 == 0)
                n_iter += 1

                # NOTE: We only use in-distribution data now. Comment this out to use ood data for rankme loss
                if use_target_rank_loss and False:
                    ood_next = next(target_rank_iterator, None)
                    if ood_next is None:
                        target_rank_iterator = iter(self.target_rank_dataloader)
                        ood_next = next(target_rank_iterator)
                    ood_images, _ = ood_next
                    ood_images = ood_images.to(self.device)

                self.optimizer.zero_grad()

                # Model optimization
                if self.fp16_precision:
                    with torch.autocast("cuda"):
                        loss, stats = self.train_iter(images, compute_stats)
                        if use_target_rank_loss:
                            loss = loss - (
                                torch.exp(self.target_rank_loss_logalpha) *
                                get_rank(self.model(ood_images))
                            )
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                else:
                    loss, stats = self.train_iter(images, compute_stats)
                    if use_target_rank_loss:
                        loss = loss - (
                            torch.exp(self.target_rank_loss_logalpha) *
                            get_rank(self.model(ood_images))
                        )
                    loss.backward()
                    self.optimizer.step()

                # Optimize to achieve target rank on OOD dataset.
                if use_target_rank_loss:
                    self.target_rank_loss_opt.zero_grad()
                    target_rank_loss = torch.exp(self.target_rank_loss_logalpha) * (
                        get_rank(self.model(ood_images)) - self.target_rank
                    ).clone().detach().requires_grad_(False)
                    target_rank_loss.backward()
                    self.target_rank_loss_opt.step()
                    stats["target_rank_loss"] = target_rank_loss.item()
                    stats["target_rank_alpha"] = torch.exp(self.target_rank_loss_logalpha).item()

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

            if self.eval_dataloader is not None:
                LOG.debug("Performing post epoch-evaluation")
                rank = rank_me(self.model, self.eval_dataloader,
                               device=self.device)
                epoch_stats.append(f"Evaluated Rank: {rank.item()}")
                mlflow.log_metric("rankme_rank", rank, step=n_iter - 1)

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
