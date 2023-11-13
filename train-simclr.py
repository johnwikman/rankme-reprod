#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from collections import deque
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.models

from torch.utils.tensorboard import SummaryWriter

import rankme_reprod
from rankme_reprod.simclr.simclr import simclr, simclr_args
from rankme_reprod.data_aug import BYOLTransform

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

LOG_LEVELS = [logging.INFO, logging.DEBUG]

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", dest="verbosity", action="count", default=0)
    parser.add_argument("-q", "--quiet", dest="log_stderr", action="store_false", help="Do not output log messages to stderr.")
    parser.add_argument("--logfile", dest="logfile", default=None, help="Output log messages to this file.")
    parser.add_argument("--tbdir", "--tensorboard-dir", dest="tensorboard_directory", default="runs")
    parser.add_argument("--dataset-dir", dest="dataset_dir", default="_datasets")
    parser.add_argument("--model-dir", dest="model_dir", default="_models")
    parser.add_argument("--device", dest="device", type=torch.device, default="cpu")
    parser.add_argument("--feat-dim", dest="featdim", type=int, default=128, help="Feature dimension")
    parser.add_argument("-j", "--workers", dest="workers", metavar="N", type=int, default=12,
                        help="number of data loading workers")
    parser.add_argument("--fp16", dest="fp16_precision", action="store_true", help="Do computations with 16-bit precision floating point (CUDA only).")

    parser << simclr_args

    args = parser.parse_args()

    # Setup the root logger
    logging.getLogger().setLevel(LOG_LEVELS[min(args.verbosity, len(LOG_LEVELS)-1)])
    LOG_FMT = logging.Formatter("[%(asctime)s %(name)s:%(lineno)d %(levelname)s]: %(message)s")
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
    writer = SummaryWriter(os.path.join(
        args.tensorboard_directory,
        datetime.now().strftime("example-simclr_%Y-%m-%d_%H.%M.%S"),
    ))

    cifar10 = torchvision.datasets.CIFAR10(args.dataset_dir,
        train=True,
        transform=BYOLTransform(crop_size=32),
        download=True,
    )

    train_loader = torch.utils.data.DataLoader(
        cifar10, batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True, drop_last=True,
    )

    # Original paper had 8192, 8192, 2048 as hidden dimensions, but we instead
    # use 2048, 2048, 1024 since that is more managable.
    encoder = torchvision.models.resnet18(num_classes=2048)

    projector_in_dim = encoder.fc.in_features
    projector = nn.Sequential(
        nn.Linear(projector_in_dim, 2048),
        nn.ReLU(), # 2048
        nn.Linear(2048, 2048),
        nn.ReLU(), # 2048
        nn.Linear(2048, 1024),
        #nn.ReLU(), # 1024
    )
    encoder.fc = nn.Identity()

    model = rankme_reprod.models.EncoderProjector(
        encoder=encoder,
        projector=projector,
    )
    model = model.to(args.device)

    optimizer = rankme_reprod.optimizers.LARS(
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

    model = simclr(args, model, optimizer, scheduler, train_loader,
        writer=writer,
        fp16_precision=args.fp16_precision,
        device=args.device,
    )
    model = model.to("cpu")

    LOG.info("SimCLR complete")

    model_path = os.path.join(args.model_dir, "simclr_resnet18.pth.tar")
    LOG.info(f"Saving model to {model_path}.")
    os.makedirs(args.model_dir, exist_ok=True)
    torch.save(model, model_path)

    LOG.info("Done. Exiting.")


if __name__ == "__main__":
    main()
