import logging
import torch
import torch.nn as nn
from tqdm import tqdm

from .pytorch_device import get_device

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


def topk_accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


@torch.no_grad()
def rank_me(model: nn.Module,
            dataloader: torch.utils.data.DataLoader,
            maxlen=None,
            device=None):
    """
    Performs the RankMe evaluation on the specified dataloader.
    """
    if maxlen is None:
        maxlen = len(dataloader)
    device = get_device(device)

    model.eval()

    LOG.debug("Doing forward pass on full dataset")
    all_outputs = []
    for images, labels in tqdm(dataloader):
        #images = torch.flatten(images, start_dim=1)
        #images = images.to(device)
        #labels = labels.to(device)
        images = images.to(device)
        output = model(images)
        all_outputs.append(output)


    model_output = torch.cat(all_outputs, dim=0)

    # model output is a matrix, now lets compute RankMe on it

    LOG.debug("Computing SVD vals")
    start_time = time.time()
    rank = get_rank(model_output)
    end_time = time.time()

    LOG.debug(f"time to compute rank: {end_time - start_time} seconds")

    return rank.cpu()


def get_rank(model_output):
    singular_values = svdvals(model_output)
    # singular values is now a vector of singular values
    # convert to distribution
    normalization_factor = torch.sum(singular_values)
    epsilon = 1e-7
    p_values = (singular_values / normalization_factor) + epsilon

    log_p_values = torch.log(p_values)

    entropy_vals = -1 * p_values * log_p_values

    rank = torch.exp(torch.sum(entropy_vals))

    return rank
