#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from datetime import datetime

import torch

from torch.utils.tensorboard import SummaryWriter

import rankme_reprod

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

LOG_LEVELS = [logging.INFO, logging.DEBUG]

parser = argparse.ArgumentParser("Example of using the SimCLR functionality")
parser.add_argument("-v", "--verbose", dest="verbosity", action="count", default=0)
parser.add_argument("-q", "--quiet", dest="log_stderr", action="store_false", help="Do not output log messages to stderr.")
parser.add_argument("--logfile", dest="logfile", default=None, help="Output log messages to this file.")
parser.add_argument("--tbdir", "--tensorboard-dir", dest="tensorboard_directory", default="runs")
parser.add_argument("--dataset-dir", dest="dataset_dir", default="_datasets")
parser.add_argument("--device", dest="device", default="cpu")

parser << rankme_reprod.simclr.simclr.simclr_args

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

# Setup TensorBoard writer
writer = SummaryWriter(os.path.join(
    args.tensorboard_directory,
    datetime.now().strftime("example-simclr_%Y-%m-%d_%H.%M.%S"),
))

"""
Unported code:
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
"""

dataset = rankme_reprod.simclr.data_aug.ContrastiveLearningDataset(args.dataset_dir)

train_dataset = dataset.get_dataset("cifar10", args.n_views)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    shuffle=True,
    num_workers=1, #args.workers,
    pin_memory=True, drop_last=True)

model = rankme_reprod.simclr.models.ResNetSimCLR(base_model="resnet18", out_dim=128) # out_dim is the number of features!

optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0,
                                                       last_epoch=-1)

s = rankme_reprod.simclr.simclr.SimCLR(
    model,
    optimizer,
    scheduler, #for what
    args=args,
    device=args.device,
)

s.train(train_loader, writer)

LOG.warning("DONE")

