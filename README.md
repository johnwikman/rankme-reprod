# Reproducability Study for RankMe (2023)
Reproducing and extending the results from RankMe (2023)


This is a change

Papers and interesting links:

* SimCLR paper: https://proceedings.mlr.press/v119/chen20j.html
* SimCLR source code: https://github.com/google-research/simclr
* A PyTorch version of SimCLR: https://github.com/sthalles/SimCLR


Install CUDA dependencies with
```sh
conda create -n rankme-cuda
conda activate rankme-cuda
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 tqdm pyyaml -c pytorch -c nvidia
conda install tensorboard
```
the environment files are insufferably slow with CUDA for some reason...
