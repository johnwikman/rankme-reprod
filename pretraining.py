#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import mlflow
import torch

from collections import deque
from datetime import datetime
from tqdm import tqdm

from src.models.encoder_projector import MODELS
#from src.pipeline.train_simclr import train_simclr
from src.utils.load_dataset import DATASETS
from src.utils.logging import init_logging
from src.models.simclr.simclr import SimCLRLoss

# Setup logger
LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())
LOG_LEVELS = [logging.INFO, logging.DEBUG]


def pretraining():
    """
    Main init script.
    Has to be wrapped in a main-method to be able to use multiprocessing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", dest="verbosity", type=int, default=0)

    parser.add_argument(
        "-q",
        "--quiet",
        dest="log_stderr",
        action="store_false",
        help="Do not output log messages to stderr.",
    )
    parser.add_argument(
        "--logfile",
        dest="logfile",
        default=None,
        help="Output log messages to this file.",
    )

    parser.add_argument(
        "--dataset",
        dest="dataset",
        choices=DATASETS.keys(),
        default=list(DATASETS.keys())[0],
        help="Dataset to train on",
    )

    parser.add_argument(
        "--model",
        dest="model",
        choices=MODELS.keys(),
        default=list(MODELS.keys())[0],
        help="Model to train",
    )

    parser.add_argument("--dataset-dir", dest="dataset_dir", default="_datasets")
    parser.add_argument("--model-dir", dest="model_dir", default="_models")
    parser.add_argument(
        "--device", dest="device", type=torch.device, default="mps"
    )  # gillar du
    parser.add_argument(
        "--feat-dim", dest="featdim", type=int, default=128, help="Feature dimension"
    )
    parser.add_argument(
        "-j",
        "--workers",
        dest="workers",
        metavar="N",
        type=int,
        default=8,
        help="number of data loading workers",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16_precision",
        action="store_true",
        help="Do computations with 16-bit precision floating point (CUDA only).",
    )

    parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        default=0.07,
        help="HYPERPARAMETER: Temperature of SimCLR",
    )

    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=64,
        help="Batch size for SimCLR",
    )

    parser.add_argument(
        "--learning-rate",
        dest="lr",
        type=float,
        default=0.1,
        help="Learning rate for SimCLR",
    )

    # add arguemnt for epochs
    parser.add_argument(
        "--epochs",
        dest="epochs",
        type=int,
        help="Number of epochs for SimCLR",
    )

    # for weight_decay
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        help="Weight decay for SimCLR",
    )

    # for n_views
    parser.add_argument(
        "--n-views",
        dest="n_views",
        type=int,
        default=2,
        help="Number of views for SimCLR",
    )

    # Parse arguments
    args = parser.parse_args()

    # Initialize arguments
    init_logging(verbosity=args.verbosity, log_stderr=args.log_stderr, logfile=args.logfile)
    LOG.debug(f"arguments: {vars(args)}")

    # insanely stupid workaround for mlflow bug
    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run():
        # Log each parameter
        # mlflow.log_param("epochs", args.epochs)
        # mlflow.log_param("featdim", args.featdim)
        # mlflow.log_param("workers", args.workers)
        # mlflow.log_param("n_views", args.n_views)
        #
        # # Hyperparams from the paper
        # # TODO Grid search these
        # mlflow.log_param("batch_size", args.batch_size)
        # mlflow.log_param("learning_rate", args.lr)
        # mlflow.log_param("temperature", args.temperature)
        # mlflow.log_param("weight_decay", args.weight_decay)

        # Call your training function

        LOG.debug(f"Loading dataset {args.dataset} with BYOL transform")
        dataset = DATASETS[args.dataset](
            dataset_dir=args.dataset_dir,
            transform=BYOLTransform(crop_size=32),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
            drop_last=True,
        )

        LOG.debug(f"Creating {args.model} model")
        model = MODELS[args.model]()
        model = model.to(args.device)

        LOG.debug("Creating LARS optimizer")
        optimizer = LARS(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay,
        )

        LOG.debug("Creating cosine annealing scheduler")
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=len(train_loader),
            eta_min=0,
            last_epoch=-1,
        )

        if args.fp16_precision:
            LOG.debug("Using 16-bit precision with CUDA")
            from torch.cuda.amp import GradScaler, autocast
            scaler = GradScaler(enabled=True)

        LOG.info(f"Start SimCLR training for {args.epochs} epochs.")

        criterion = SimCLRLoss(args.temperature).to(args.device)

        n_iter = 0
        for epoch_counter in range(self.epochs):
            for (img_x, img_y), _ in tqdm(train_loader):
                img_x = img_x.to(args.device)
                img_y = img_y.to(args.device)
                self.optimizer.zero_grad()

                if args.fp16_precision:
                    with autocast(enabled=True):
                        loss = criterion(model(img_x), model(img_y))
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                else:
                    loss = criterion(model(img_x), model(img_y))
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


        LOG.info("SimCLR complete")
        model_path = os.path.join(args.model_dir, "simclr_resnet18.pth.tar")
        LOG.info(f"Saving model to {model_path}.")
        os.makedirs(args.model_dir, exist_ok=True)
        model.save(model_path)

        LOG.info("Done. Exiting.")
        mlflow.end_run()


if __name__ == "__main__":
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:8080"
    pretraining()
