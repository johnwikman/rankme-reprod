#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import torchvision
import torchvision.models

from collections import deque
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import src
from src.models.vicreg import VICRegLoss
from src.utils.data_aug import BYOLTransform
from src.models.encoder_projector import EncoderProjector

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


if __name__ == "__main__":
    LOG_LEVELS = [logging.INFO, logging.DEBUG]

    DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("Example of using the SimCLR functionality")
    parser.add_argument("-v", "--verbose", dest="verbosity", action="count", default=0)
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
    parser.add_argument("--dataset-dir", dest="dataset_dir", default="_datasets")
    parser.add_argument("--device", dest="device", type=torch.device, default=DEFAULT_DEVICE)
    parser.add_argument(
        "--feat-dim", dest="featdim", type=int, default=128, help="Feature dimension"
    )
    parser.add_argument(
        "-j",
        "--workers",
        dest="workers",
        metavar="N",
        type=int,
        default=12,
        help="number of data loading workers",
    )

    args = parser.parse_args()

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

    train_dataset = torchvision.datasets.CIFAR10(
        args.dataset_dir,
        train=True,
        transform=BYOLTransform(crop_size=32),
        download=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    # The output from ResNet is the number of features
    resnet = torchvision.models.resnet18(num_classes=args.featdim)
    resnet_outdim = resnet.fc.in_features
    resnet.fc = nn.Identity()

    model = EncoderProjector(
        encoder=resnet,
        projector=nn.Sequential(
            nn.Linear(resnet_outdim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
        )
    )

    encoder = model.to(args.device)

    lr = 1e-3
    wd = 1e-6
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=len(train_loader),
        eta_min=0.0001 * lr,
        last_epoch=-1,
    )

    LOG.debug("DO A TRAINING LOOP HERE")
    losses = deque(maxlen=50)
    criterion = VICRegLoss()
    for epoch in range(10):
        LOG.info(f"Epoch {epoch}")
        train_iterator = tqdm(train_loader)
        for images, _ in train_iterator:
            assert len(images) == 2
            opt.zero_grad()
            x = model(images[0])
            y = model(images[1])
            loss = criterion(x, y)
            loss.backward()
            opt.step()
            losses.append(loss.item())
            train_iterator.set_postfix_str(f"avg loss: {sum(losses)/len(losses)}")
