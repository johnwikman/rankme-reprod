# Reproducability Study for RankMe (2023)
Reproducing and extending the results from RankMe (2023)

Papers and interesting links:

* SimCLR paper: https://proceedings.mlr.press/v119/chen20j.html
* SimCLR source code: https://github.com/google-research/simclr
* A PyTorch version of SimCLR: https://github.com/sthalles/SimCLR

# How to?

The [main](main.py) script is responsible for communication with the package contained in `src`.
We have spread out default values all over the code to make a deterministic process feel more like a black box. Enjoy!

Do like this:
```bash
python main.py --verbose --device=cuda --epochs=5
```

## Dependencies

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

So in effect we need to test for different pretrain models, using different hyperparams, using different OOD datasets


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


