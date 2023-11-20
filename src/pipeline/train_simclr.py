#!/usr/bin/env python3

import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
import torchvision.models

from torch.utils.tensorboard import SummaryWriter
from collections import deque
from datetime import datetime

from ..models.simclr.simclr import SimCLR
from ..utils.data_aug import BYOLTransform
from ..utils.pytorch_device import get_device
from ..models import encoder_projector
from ..utils.optimizers import LARS

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

LOG_LEVELS = [logging.INFO, logging.DEBUG]


def train_simclr(parser):
    print("Args received in train_simclr: ", parser)
    args = parser.parse_args()

    device = get_device(args.device)

    # Setup the root logger
    logging.getLogger().setLevel(LOG_LEVELS[min(args.verbosity, len(LOG_LEVELS) - 1)])
    LOG_FMT = logging.Formatter(
        "[%(asctime)s %(name)s:%(lineno)d %(levelname)s]: %(message)s"
    )
    if args.log_stderr:
        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setFormatter(LOG_FMT)
        logging.getLogger().addHandler(stderr_handler)
    if args.logfile is not None:
        logfile_handler = logging.FileHandler(args.logfile)
        logfile_handler.setFormatter(LOG_FMT)
        logging.getLogger().addHandler(logfile_handler)

    LOG.debug(f"arguments: {vars(args)}")

    # Setup TensorBoard writer
    # TODO: refactor to MLFlow
    writer = SummaryWriter(
        os.path.join(
            args.tensorboard_directory,
            datetime.now().strftime("example-simclr_%Y-%m-%d_%H.%M.%S"),
        )
    )

    cifar10 = torchvision.datasets.CIFAR10(
        args.dataset_dir,
        train=True,
        transform=BYOLTransform(crop_size=32),
        download=True,
    )

    train_loader = torch.utils.data.DataLoader(
        cifar10,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

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

    model = encoder_projector.EncoderProjector(
        encoder=encoder,
        projector=projector,
    )

    model = model.to(device)

    optimizer = LARS(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-6,
    )

    # Cosine annealing scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=len(train_loader),
        eta_min=0,
        last_epoch=-1,
    )

    LOG.info("Starting SimCLR training")

    model = SimCLR(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        writer=writer,
        fp16_precision=args.fp16_precision,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        temperature=args.temperature,
        n_views=args.n_views,
        device=device,
    )

    model.train(train_loader)

    LOG.info("SimCLR complete")

    model_path = os.path.join(args.model_dir, "simclr_resnet18.pth.tar")
    LOG.info(f"Saving model to {model_path}.")
    os.makedirs(args.model_dir, exist_ok=True)
    model.save(model_path)

    LOG.info("Done. Exiting.")
