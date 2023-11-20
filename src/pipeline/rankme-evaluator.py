'''
This file contains the pipeline for producing a RankMe evaluation of a model.

It takes as input a path to a model, as well as a dataset to evaluate on.

It outputs the RankMe evaluation of the model, saving it to a file in the same folder as the model.
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import pickle
import os
import numpy as np
from tqdm import tqdm
from torch.linalg import svdvals
from rankme_reprod.simclr.data_aug import simclr_transform
from torch.utils.data import DataLoader, Dataset, TensorDataset


def rank_me(model_path, dataset_path):

    model = torch.load(model_path)

    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    #dataset, labels = load_cifar10(dataset_path)

    simclr_train_dataset = torchvision.datasets.CIFAR10(dataset_path,
        train=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
        download=True,
    )

    dataloader = DataLoader( ## NOTE: some hardcoded values here
        simclr_train_dataset, batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True, drop_last=True,
    )


    # HERE BE FORWARD PASS
    all_outputs = []
    with torch.no_grad():
        
        for batch in tqdm(dataloader):
            images, labels = batch
            #images = torch.flatten(images, start_dim=1)
            #images = images.to(device)
            #labels = labels.to(device)
            output = model(images)
            all_outputs.append(output)


    model_output = torch.cat(all_outputs, dim=0)
    
    
    # model output is a matrix, now lets compute RankMe on it

    rank = get_rank(model_output)

    return rank

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

if __name__ == '__main__':
    # read in a basic dataset, create a basic model, and run rankme on it

    dataset_path = "_datasets"

    test_model_path = "_models/simclr_resnet18.pth.tar"

    rank = rank_me(test_model_path, dataset_path)

    print("rank of model", rank)
