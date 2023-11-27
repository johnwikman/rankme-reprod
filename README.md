# Reproducability Study for RankMe (2023)
Reproducing and extending the results from RankMe (2023)

Papers and interesting links:

* SimCLR paper: https://proceedings.mlr.press/v119/chen20j.html
* SimCLR source code: https://github.com/google-research/simclr
* A PyTorch version of SimCLR: https://github.com/sthalles/SimCLR

# How to?

## MLFlow

### 1. Install MLFlow

```sh
pip install mlflow
```

### 2. Start MLFlow server

To set it up locally, run the following command in the root of the project:

```sh
# pick a favorite unused port
mlflow server --host 127.0.0.1 --port 8080

# kanske såhär
mlflow server --backend-store-uri postgres://avnadmin:AVNS_3gVxQDkJiI0AmDebBM3@rankme-reprod-rankme.a.aivencloud.com:12005/defaultdb?sslmode=require

```

Later, we will setup a URI to an external database. Very mumma!

### 3. Run the code
The repo contains a [MLproject](MLproject) file which defines the entry points for the project. First time you run it, it will create a conda environment corresponding to the name at the top of the file. 

To run the code, use the following command:

```sh
# -P for "parameter"
mlflow run . -P device="mps"

# vill man ha fler parameterar får man lägga till fler -P
mlflow run . -P device="mps" -P epochs=100 -P batch_size=64 # osv

# om John måste köra den:
mlflow run . -P device="cuda" -P workers=16 --env-manager local
```

Currently, all parameters have default values except `device`. These values are logged as "parameters" within an MLFlow run. In the experiments, we log "metrics", which are running values that change over time.

### 4. View the results

To view the results, go to [localhost:8080](http://localhost:8080) in your browser. You should see a list of experiments. Click on one of them to see the results.

## TODO

* Fix remote database
* Fix pipeline. Can be done with MLFlow.
* Perhaps `main.py` is responsible for managing pretrain and validate? I believe that makes sense, but I am also n00b.



# Dependencies

> **NOTE:** Outdated, going to fix MLFlow our savior.


Install CUDA dependencies with
```sh
conda create -n rankme
conda activate rankme
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 tqdm pyyaml -c pytorch -c nvidia
conda install tensorboard
```
the environment files are insufferably slow with CUDA for some reason...


Example uses:
```
./example-simclr.py -v --device=cuda -j16 --feat-dim=256 --epochs 100
```

### Pipeline
<summary> 
    <details>Click to expand</details>

    1. Choose a model for pretraining (SimCLR, VicReg) (Freedom of model)
2. Choose hyperparameters for pretraining algorithm (matching paper) (Freedom of hyperparams)
3. Choose a dataset to pretrain with (In theory freedom of dataset, but in practice maybe always Tiny ImageNet)
4. Attatch a model head (MLP with dimensions matching the paper and output matching pretrain dataset) (No freedom)
5. Pretrain the model with given hyperparameters (No freedom)
6. Save pretrained model
(P1 complete)
7. Select an OOD dataset (CIFAR100, iNaturalist mini, SUN if we have time) (Freedom of dataset)
(From here on there is no freedom)
8. Compute the RankMe evaluation of the model on the OOD dataset **eventuellt efterat**
9. Store RankMe evaluation with the model
(P2 complete)
10. Attatch a new, linear projector, with output dimensions matching the OOD dataset
11. Freeze the encoder
12. Train the model projector for some number of epochs (take hyperparams from article)
13. Save the finetuned model together with the pretrained model
14. Extract accuracy on OOD dataset
15. Store accuracy together with prev data
16. (Optionally) Store all the results from pipeline 1-3 on some remote database
(P3 complete)
17. Extract all results, specifically rankme eval and accuracy
18. Produce plots of the results, do any data analysis you want to include in results.

So in effect we need to test for different pretrain models, using different hyperparams, using different OOD datasets.
</summary>



### LOCAL STRUCTURE
INPUT FOLDERS:
- Linus fixar
OUTPUT FOLDERS:

- (DIR)all_results
--- (DIR)SimCLR
----- (DIR)(different hyperparams)
------- PRETRAINED_MODEL
------- RANKME_EVAL
------- (DIR) CIFAR100
--------- FINETUNED_MODEL
--------- ACCURACY
------- (DIR) iNaturalist Mini
--------- FINETUNED_MODEL
--------- ACCURACY
--- (DIR)VicReg
----- (DIR)(different hyperparams)
------- PRETRAINED_MODEL
------- RANKME_EVAL
------- (DIR) CIFAR100
--------- FINETUNED_MODEL
--------- ACCURACY
------- (DIR) iNaturalist Mini
--------- FINETUNED_MODEL
--------- ACCURACY


