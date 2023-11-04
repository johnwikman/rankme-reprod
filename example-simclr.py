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

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import rankme_reprod
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


simclr_train_dataset = torchvision.datasets.CIFAR10(args.dataset_dir,
    train=True,
    transform=simclr_transform(32, args.n_views),
    download=True,
)

simclr_train_loader = torch.utils.data.DataLoader(
    simclr_train_dataset, batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True, drop_last=True,
)

# The output from ResNet is the number of features
encoder = torchvision.models.resnet18(num_classes=args.featdim)
# Add extra fully connected layer to model
mlpdim = encoder.fc.in_features
encoder.fc = nn.Sequential(nn.Linear(mlpdim, mlpdim), nn.ReLU(), encoder.fc)

# JIT Compile the encoder model
#encoder = torch.jit.script(encoder)

encoder = encoder.to(args.device)

encoder_optimizer = torch.optim.Adam(encoder.parameters(), args.lr, weight_decay=args.weight_decay)

encoder_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    encoder_optimizer,
    T_max=len(simclr_train_loader),
    eta_min=0,
    last_epoch=-1,
)

simclr(
    args,
    encoder,
    encoder_optimizer,
    encoder_scheduler,
    simclr_train_loader,
    writer=writer,
    fp16_precision=args.fp16_precision,
    device=args.device,
)

LOG.info("SimCLR complete, now optimizing on supervised training")

model = rankme_reprod.models.LatentClassifier(
    encoder=encoder,
    projector=nn.Sequential(
        nn.Linear(args.featdim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10),
    ),
)
model = model.to(args.device)
opt = torch.optim.Adam(
    model.parameters(),
    #model.projector.parameters(), # optimize only the projector network
    args.lr,
    weight_decay=args.weight_decay,
)


cifar10_train = torchvision.datasets.CIFAR10(
    args.dataset_dir,
    transform=torchvision.transforms.ToTensor(),
    train=True,
    download=True,
)

cifar10_trainloader = torch.utils.data.DataLoader(
    cifar10_train,
    #simclr_train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.workers,
    pin_memory=True,
    drop_last=True,
)

criterion = nn.CrossEntropyLoss().to(args.device)
train_iterator = tqdm(cifar10_trainloader)
for _ in range(args.epochs):
    top1_accuracies = []
    top5_accuracies = []
    losses = []
    for images, labels in train_iterator:
        images = images.to(args.device)
        labels = labels.to(args.device)

        opt.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        opt.step()
        with torch.no_grad():
            top1, top5 = rankme_reprod.evaluate.topk_accuracy(logits, labels, topk=(1, 5))
            top1_accuracies.append(float(top1[0]))
            top5_accuracies.append(float(top5[0]))
            losses.append(float(loss.item()))
            train_iterator.set_postfix_str(" | ".join([
                f"avg. loss: {sum(losses[-10:]) / len(losses[-10:]):.05f}",
                f"top 1 acc: {sum(top1_accuracies[-10:]) / len(top1_accuracies[-10:]):.05f}",
                f"top 5 acc: {sum(top5_accuracies[-10:]) / len(top5_accuracies[-10:]):.05f}",
            ]))

    LOG.info(f"Average loss: {sum(losses) / len(losses)}")
    LOG.info(f"Top 1 accuracy: {sum(top1_accuracies) / len(top1_accuracies)}")
    LOG.info(f"Top 5 accuracy: {sum(top5_accuracies) / len(top5_accuracies)}")
