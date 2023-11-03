#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.models

from torch.utils.tensorboard import SummaryWriter

from rankme_reprod.simclr.simclr import simclr, simclr_args
from rankme_reprod.simclr.data_aug import simclr_transform

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

LOG_LEVELS = [logging.INFO, logging.DEBUG]

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser("Example of using the SimCLR functionality")
parser.add_argument("-v", "--verbose", dest="verbosity", action="count", default=0)
parser.add_argument("-q", "--quiet", dest="log_stderr", action="store_false", help="Do not output log messages to stderr.")
parser.add_argument("--logfile", dest="logfile", default=None, help="Output log messages to this file.")
parser.add_argument("--tbdir", "--tensorboard-dir", dest="tensorboard_directory", default="runs")
parser.add_argument("--dataset-dir", dest="dataset_dir", default="_datasets")
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


train_dataset = torchvision.datasets.CIFAR10(args.dataset_dir,
    train=True,
    transform=simclr_transform(32, args.n_views),
    download=True,
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True, drop_last=True,
)

# The output from ResNet is the number of features
model = torchvision.models.resnet18(num_classes=args.featdim)
# Add extra fully connected layer to model
mlpdim = model.fc.in_features
model.fc = nn.Sequential(nn.Linear(mlpdim, mlpdim), nn.ReLU(), model.fc)

# JIT Compile the model
model = torch.jit.script(model)

model = model.to(args.device)

optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                       last_epoch=-1)

simclr(
    args,
    model,
    optimizer,
    scheduler,
    train_loader,
    writer=writer,
    device=args.device,
)

LOG.info("DONE")
