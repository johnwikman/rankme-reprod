import logging
import os
import sys

import torch
import torch.nn.functional as F
#from torch.cuda.amp import GradScaler, autocast
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from .utils import save_config_file, accuracy, save_checkpoint

#torch.manual_seed(0)

"""
parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-dataset-name', default='stl10',
                    help='dataset name', choices=['stl10', 'cifar10'])
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.0003, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out_dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
"""

from ..method_args import MethodArguments, MethodArg as MA


LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


simclr_args = MethodArguments(
    epochs = MA("--epochs", default=200, type=int, metavar="N",
                help="number of total epochs to run"),
    batch_size = MA("-b", "--batch-size", default=256, type=int, metavar="N",
                    help="mini-batch size (default: 256), this is the total "
                         "batch size of all GPUs on the current node when "
                         "using Data Parallel or Distributed Data Parallel"),
    lr = MA("--lr", "--learning-rate", default=0.0003, type=float,
            metavar="LR", help="initial learning rate"),
    weight_decay = MA("--wd", "--weight-decay", default=1e-4, type=float,
                      metavar="W", help="weight decay (default: 1e-4)"),
    temperature = MA("--temperature", default=0.07, type=float,
                     help="softmax temperature (default: 0.07)"),
    n_views = MA("--n-views", default=2, type=int, metavar="N",
                 help="Number of views for contrastive learning training."),
)



class SimCLR(object):
    def __init__(self, model, optimizer, scheduler,
                 args=simclr_args.defaults,
                 device="cpu"):
        self.args = args
        #self.args = kwargs['args']
        self.model = model #kwargs['model'].to(self.args.device)
        self.optimizer = optimizer #kwargs['optimizer']
        self.scheduler = scheduler #kwargs['scheduler']
        self.device = device
        #self.writer = SummaryWriter()
        #logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, writer):

        #scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(writer.log_dir, self.args)

        n_iter = 0
        LOG.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        #LOG.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            for images, _ in tqdm(train_loader):
                images = torch.cat(images, dim=0)

                images = images.to(self.device)

                #with autocast(enabled=self.args.fp16_precision):
                features = self.model(images)
                logits, labels = self.info_nce_loss(features)
                loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                #scaler.scale(loss).backward()
                loss.backward()

                #scaler.step(self.optimizer)
                #scaler.update()
                self.optimizer.step()

                if n_iter % 100 == 0: #self.args.log_every_n_steps == 0:
                    top1, top5 = accuracy(logits, labels, topk=(1, 5))
                    writer.add_scalar('loss', loss, global_step=n_iter)
                    writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    writer.add_scalar('learning_rate', self.scheduler.get_lr()[0], global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()
            LOG.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")

        LOG.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            "epoch": self.args.epochs,
            "arch": "resnet18", #self.args.arch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(writer.log_dir, checkpoint_name))
        LOG.info(f"Model checkpoint and metadata has been saved at {writer.log_dir}.")
