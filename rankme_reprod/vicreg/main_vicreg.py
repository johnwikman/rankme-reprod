# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from pathlib import Path
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torch.distributed as dist
import torchvision.datasets as datasets

#import augmentations as aug
#from distributed import init_distributed_mode

#import resnet


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)

    # Data
    parser.add_argument("--data-dir", type=Path, default="/path/to/imagenet", required=True,
                        help='Path to the image net dataset')

    # Checkpoints
    parser.add_argument("--exp-dir", type=Path, default="./exp",
                        help='Path to the experiment folder, where all logs/checkpoints will be stored')
    parser.add_argument("--log-freq-time", type=int, default=60,
                        help='Print logs to the stats.txt file every [log-freq-time] seconds')

    # Model
    parser.add_argument("--arch", type=str, default="resnet50",
                        help='Architecture of the backbone encoder network')
    parser.add_argument("--mlp", default="8192-8192-8192",
                        help='Size and number of layers of the MLP expander head')

    # Optim
    parser.add_argument("--epochs", type=int, default=100,
                        help='Number of epochs')
    parser.add_argument("--batch-size", type=int, default=2048,
                        help='Effective batch size (per worker batch size is [batch-size] / world-size)')
    parser.add_argument("--base-lr", type=float, default=0.2,
                        help='Base learning rate, effective learning after warmup is [base-lr] * [batch-size] / 256')
    parser.add_argument("--wd", type=float, default=1e-6,
                        help='Weight decay')

    # Loss
    parser.add_argument("--sim-coeff", type=float, default=25.0,
                        help='Invariance regularization loss coefficient')
    parser.add_argument("--std-coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov-coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')

    # Running
    parser.add_argument("--num-workers", type=int, default=10)
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    # Distributed
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):
    # This is all parallel efficiency wish wash
    #torch.backends.cudnn.benchmark = True
    #init_distributed_mode(args)
    #print(args)
    #gpu = torch.device(args.device)

    #if args.rank == 0:
    #    args.exp_dir.mkdir(parents=True, exist_ok=True)
    #    stats_file = open(args.exp_dir / "stats.txt", "a", buffering=1)
    #    print(" ".join(sys.argv))
    #    print(" ".join(sys.argv), file=stats_file)

    #transforms = aug.TrainTransform()

    # NOTE: We handle this elsewhere
    #dataset = datasets.ImageFolder(args.data_dir / "train", transforms)
    #sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
    #assert args.batch_size % args.world_size == 0
    #per_device_batch_size = args.batch_size // args.world_size
    #loader = torch.utils.data.DataLoader(
    #    dataset,
    #    batch_size=per_device_batch_size,
    #    num_workers=args.num_workers,
    #    pin_memory=True,
    #    sampler=sampler,
    #)

    # OK: so "model" is the VicReg algorithm
    model = VICReg(args).cuda(gpu)
    # This does what?
    # https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html#torch.nn.SyncBatchNorm.convert_sync_batchnorm
    # This just seems to take the batch norm over multiple processes, splitting the batch in
    #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # This is also a data parallel thing
    # https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html#distributeddataparallel
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # We already have this...
    #optimizer = LARS(
    #    model.parameters(),
    #    lr=0,
    #    weight_decay=args.wd,
    #    weight_decay_filter=exclude_bias_and_norm,
    #    lars_adaptation_filter=exclude_bias_and_norm,
    #)

    # This just tries to load a checkpoint
    #if (args.exp_dir / "model.pth").is_file():
    #    if args.rank == 0:
    #        print("resuming from checkpoint")
    #    ckpt = torch.load(args.exp_dir / "model.pth", map_location="cpu")
    #    start_epoch = ckpt["epoch"]
    #    model.load_state_dict(ckpt["model"])
    #    optimizer.load_state_dict(ckpt["optimizer"])
    #else:
    #    start_epoch = 0

    # Some training loop here?
    #start_time = last_logging = time.time()
    # Scaler we can have later, probably don't need it initially...
    #scaler = torch.cuda.amp.GradScaler()
    for epoch in range(start_epoch, args.epochs):
        # Data distributed
        #sampler.set_epoch(epoch)
        for step, ((x, y), _) in enumerate(loader, start=epoch * len(loader)):
            x = x.cuda(gpu, non_blocking=True)
            y = y.cuda(gpu, non_blocking=True)

            # LR scheduler instead? Is this some funky LR schedule?
            lr = adjust_learning_rate(args, optimizer, loader, step)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                loss = model.forward(x, y)
            # Replace the scaler with the simple loss.backward()
            #scaler.scale(loss).backward()
            #scaler.step(optimizer)
            #scaler.update()
            loss.backward()
            optimizer.step()

            # This is just logging stuff, not important
            #current_time = time.time()
            #if args.rank == 0 and current_time - last_logging > args.log_freq_time:
            #    stats = dict(
            #        epoch=epoch,
            #        step=step,
            #        loss=loss.item(),
            #        time=int(current_time - start_time),
            #        lr=lr,
            #    )
            #    print(json.dumps(stats))
            #    print(json.dumps(stats), file=stats_file)
            #    last_logging = current_time
        # args.rank is just a check to see that we are in the main process
        #this just checkpoints the model, saves it
        #if args.rank == 0:
        #    state = dict(
        #        epoch=epoch + 1,
        #        model=model.state_dict(),
        #        optimizer=optimizer.state_dict(),
        #    )
        #    torch.save(state, args.exp_dir / "model.pth")
    # This saves the model at the end
    #if args.rank == 0:
    #    torch.save(model.module.backbone.state_dict(), args.exp_dir / "resnet50.pth")


# This is cosine annealing LR with 10 epochs of warmup
# This exists as CosineAnnealingLR(T_max=len(loader), eta_min=base_lr*0.001)
# The base_lr is implicit from the underlying optimizer.
def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.base_lr * args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


class VICReg(nn.Module):
    def __init__(self,
                 sim_coeff=25.0,
                 std_coeff=25.0,
                 cov_coeff=1.0,
                 batch_size=512):
        super().__init__()
        #self.args = args
        self.sim_coeff = 25.0
        self.std_coeff = 25.0
        self.cov_coeff = 1.0
        self.batch_size = batch_size
        #self.num_features = int(args.mlp.split("-")[-1])
        #self.backbone, self.embedding = resnet.__dict__[args.arch](
        #    zero_init_residual=True
        #)
        #self.projector = Projector(args, self.embedding)

    def forward(self, x, y):
        # So this is the algorithm:
        #  1. do one forward pass in the network for each augmentation,
        #  2. compute the mean square error between the outputs
        #  3. convert the batches such that they have µ=0
        #  4. compute the sample standard deviation with µ=0
        #  5. compute [sqrt(ReLU(1-std_x)) + sqrt(ReLU(1-std_y))] / 2
        #  6. compute the covariance loss
        #  7. the returned loss is a weighted sum of the covariance, std, and mse loss.
        x = self.projector(self.backbone(x))
        y = self.projector(self.backbone(y))

        repr_loss = F.mse_loss(x, y)

        # This only collects gradients from different processes, and stacks them together.
        #x = torch.cat(FullGatherLayer.apply(x), dim=0)
        #y = torch.cat(FullGatherLayer.apply(y), dim=0)
        # This appears to make sure that it has mean 0 in both x and y
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_y = torch.sqrt(y.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

        cov_x = (x.T @ x) / (self.args.batch_size - 1)
        cov_y = (y.T @ y) / (self.args.batch_size - 1)
        cov_loss = off_diagonal(cov_x).pow_(2).sum().div(
            self.num_features
        ) + off_diagonal(cov_y).pow_(2).sum().div(self.num_features)

        loss = (
            self.sim_coeff * repr_loss
            + self.std_coeff * std_loss
            + self.cov_coeff * cov_loss
        )
        return loss


def Projector(args, embedding):
    mlp_spec = f"{embedding}-{args.mlp}"
    layers = []
    f = list(map(int, mlp_spec.split("-")))
    for i in range(len(f) - 2):
        layers.append(nn.Linear(f[i], f[i + 1]))
        layers.append(nn.BatchNorm1d(f[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(f[-2], f[-1], bias=False))
    return nn.Sequential(*layers)

# This is LARS stuff
#def exclude_bias_and_norm(p):
#    return p.ndim == 1


def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

# Was maybe used somewhere...
#def batch_all_gather(x):
#    x_list = FullGatherLayer.apply(x)
#    return torch.cat(x_list, dim=0)


# This only collects values and gradients from multiple processes,
class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.environ["SLURM_JOB_ID"]}')
    exit()


def handle_sigterm(signum, frame):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VICReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
