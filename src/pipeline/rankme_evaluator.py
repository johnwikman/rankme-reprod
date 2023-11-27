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
import torchvision.datasets as datasets
import pickle
import os
import numpy as np
from tqdm import tqdm
from torch.linalg import svdvals
#from src.utils.data_aug import simclr_transform
from torch.utils.data import DataLoader, Dataset, TensorDataset
from src.utils.load_dataset import load_dataset
import time





def rank_me(model: nn.Module, dataset: str, device="cpu"):
    '''
    Input: 
    model_path (should pretty much always be _models/{name of saved model})
    dataset_path (should pretty much always be _datasets)
    ood_dataset (either 'CIFAR100' or 'iNaturalist')
    device (device to compute on, default is cpu, can also pass cuda or MPS)

    returns:
    rank of the model on the ood dataset

    '''

    #model = torch.load(model_path)

    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.to(device)

    #train_dataset = load_dataset(dataset_path, dataset_name=ood_dataset)
    train_dataset = dataset

    dataloader = DataLoader( ## NOTE: some hardcoded values here
        train_dataset, batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True, drop_last=True,
    )

    


    # HERE BE FORWARD PASS
    print("Doing forward pass on full dataset")
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

    start_time = time.time()
    rank = get_rank(model_output)
    end_time = time.time()

    print("time to compute rank", end_time - start_time)

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

    target_dataset = "CIFAR100"

    rank = rank_me(test_model_path, dataset_path, target_dataset)

    print("rank of model", rank)
