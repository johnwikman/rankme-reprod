#!/usr/bin/env python3

import argparse
import logging
import os
import sys
import mlflow
import torch
import torchvision

from collections import deque
from datetime import datetime
from tqdm import tqdm

from src.models.encoder_projector import MODELS
from src.models import SimCLR, VICReg
from src.utils.load_dataset import DATASETS
from src.utils.logging import init_logging
from src.utils.data_aug import BYOLTransform, SplitTransform
from src.utils.optimizers import LARS
from src.utils.pytorch_device import get_device

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
        "--eval-dataset",
        dest="eval_dataset",
        choices=DATASETS.keys(),
        default=None,
        help="Dataset to evaluate on during training",
    )

    parser.add_argument(
        "--trainer",
        dest="trainer",
        choices={"simclr", "vicreg"},
        default="simclr",
        help="The pretraining algorithm to use",
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
        "--device", dest="device", type=get_device, default=get_device(None)
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
        "--sim-coeff",
        dest="sim_coeff",
        type=float,
        default=25.0,
        help="HYPERPARAMETER: Similarity coefficient of VICReg",
    )

    parser.add_argument(
        "--std-coeff",
        dest="std_coeff",
        type=float,
        default=25.0,
        help="HYPERPARAMETER: Standard deviation coefficient of VICReg",
    )

    parser.add_argument(
        "--cov-coeff",
        dest="cov_coeff",
        type=float,
        default=1.0,
        help="HYPERPARAMETER: Covariance coefficient of VICReg",
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

    parser.add_argument("--use-target-rank", dest="use_target_rank", type=int, default=0,
                        help="Optimize for target rank (default: no target rank)")
    parser.add_argument("--target-rank", dest="target_rank", type=float, default=None,
                        help="Target rank to optimize for")
    parser.add_argument("--target-rank-logalpha", dest="target_rank_logalpha", type=float, default=0.2,
                        help="Initial value for target rank logalpha")
    parser.add_argument("--target-rank-lr", dest="target_rank_lr", type=float, default=1e-4,
                        help="Optimizer rate for target rank logalpha")

    # Parse arguments
    args = parser.parse_args()

    # Initialize arguments
    init_logging(verbosity=args.verbosity, log_stderr=args.log_stderr, logfile=args.logfile)
    LOG.debug(f"arguments: {vars(args)}")

    # insanely stupid workaround for mlflow bug
    if mlflow.active_run():
        mlflow.end_run()

    run_name = datetime.now().strftime(f"run_%Y%m%d_%H%M%S_{args.trainer}")
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("mlflow.runName", run_name)
        LOG.debug(f"Loading dataset {args.dataset} with BYOL transform")
        dataset = DATASETS[args.dataset](
            dataset_path=args.dataset_dir,
            transform=SplitTransform(
                # Second transformation is used to probe the original image for the rankme loss
                t1=BYOLTransform(crop_size=32),
                t2=torchvision.transforms.ToTensor(),
            )
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
            T_max=len(dataloader),
            eta_min=0,
            last_epoch=-1,
        )

        common_kwargs = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "fp16_precision": args.fp16_precision,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "device": args.device,
        }

        if args.use_target_rank != 0:
            target_rank_logalpha = torch.nn.Parameter(torch.tensor(
                args.target_rank_logalpha,
                device=args.device,
                dtype=torch.float32,
            ))
            target_rank_loss_opt = torch.optim.Adam(
                [target_rank_logalpha],
                lr=args.target_rank_lr
            )
            common_kwargs["use_target_rank_loss"] = True
            common_kwargs["target_rank"] = args.target_rank
            common_kwargs["target_rank_loss_logalpha"] = target_rank_logalpha
            common_kwargs["target_rank_loss_opt"] = target_rank_loss_opt

        if args.trainer == "simclr":
            trainer = SimCLR(
                temperature=args.temperature,
                **common_kwargs
            )
        elif args.trainer == "vicreg":
            trainer = VICReg(
                sim_coeff=args.sim_coeff,
                std_coeff=args.std_coeff,
                cov_coeff=args.cov_coeff,
                **common_kwargs
            )
        else:
            raise ValueError(f"Unsupported argument: {args.vicreg}")

        if args.eval_dataset is not None:
            eval_dataset = DATASETS[args.eval_dataset](
                dataset_path=args.dataset_dir,
                transform=torchvision.transforms.ToTensor(),
            )
            eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.workers,
                pin_memory=True,
                drop_last=True,
            )
            trainer.set_eval_dataloader(eval_dataloader)

        trainer.train(dataloader)

        model_path = os.path.join(args.model_dir, f"{args.trainer}_{args.model}.pth.tar")
        os.makedirs(args.model_dir, exist_ok=True)
        LOG.info(f"Saving model to {model_path}.")
        trainer.save(model_path)

        mlflow.end_run()


if __name__ == "__main__":
    if "MLFLOW_TRACKING_URI" not in os.environ:
        raise EnvironmentError("Environment variable MLFLOW_TRACKING_URI not "
                               "set. Example: set it with\n"
                               "export MLFLOW_TRACKING_URI=\"http://127.0.0.1:8080\"")
    pretraining()
