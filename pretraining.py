#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import mlflow
import torch

from collections import deque
from datetime import datetime

from src.pipeline.train_simclr import train_simclr

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
        train_simclr(parser)
        mlflow.end_run()


if __name__ == "__main__":
    pretraining()
