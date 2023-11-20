#!/usr/bin/env python3

import argparse
import logging
import os
import sys

import torch

from torch.utils.tensorboard import SummaryWriter
from collections import deque
from datetime import datetime

from src.pipeline.train_simclr import train_simclr

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

LOG_LEVELS = [logging.INFO, logging.DEBUG]


def main():
    """
    Main init script.
    Has to be wrapped in a main-method to be able to use multiprocessing.
    """
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        "--tbdir", "--tensorboard-dir", dest="tensorboard_directory", default="runs"
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
        default=10,
        help="Number of epochs for SimCLR",
    )

    # for weight_decay
    parser.add_argument(
        "--weight-decay",
        dest="weight_decay",
        type=float,
        default=1e-6,
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

    # parser << simclr_args

    train_simclr(parser)


if __name__ == "__main__":
    main()
