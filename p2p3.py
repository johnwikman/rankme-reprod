import argparse
import mlflow
import torch
import os
import logging



# import path
workingdir = os.getcwd()
#print(workingdir)
#os.chdir(workingdir)
#os.chdir("../")
import src.pipeline.rankme_evaluator as RankMe
import src.pipeline.finetune as finetune
import src.pipeline.finetune_caltech as finetune_caltech
from src.models.encoder_projector import EncoderProjector
from torchvision.transforms import transforms
import pandas as pd
from src.utils.logging import init_logging
from src.utils.pytorch_device import get_device
from src.utils.evaluate import rank_me
from src.utils.load_dataset import load_dataset, DATASETS
from src.utils.caltech import get_caltech_loaders
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())

dataset_path = "_datasets/"

#device = "cpu"
#finetune_epochs = 2

def main():
    parser = argparse.ArgumentParser("Evaluate rankme")
    parser.add_argument("--dataset", dest="dataset", choices=DATASETS.keys(), default="cifar100", help="Dataset to evaluate on.")
    parser.add_argument("--device", dest="device", type=get_device, default=get_device(None), help="Device to run on.")
    parser.add_argument("--epochs", dest="finetune_epochs", default=10, type=int, help="Epochs to finetune on")
    parser.add_argument("-j", "--num-workers", dest="num_workers", default=4, type=int, help="Number of workers to use for the dataloaders.")

    args = parser.parse_args()

    init_logging(verbosity=1, log_stderr=True)

    all_runs = mlflow.search_runs()
    # filter out runs where status is failed
    all_successful_runs = all_runs[all_runs['status'] == 'FINISHED']

    # NOTE: add iNaturalist to this filter later
    client = mlflow.tracking.MlflowClient()

    cifar_transform = transforms.Compose([
            transforms.Resize(224),  # Resize to 224x224
            transforms.ToTensor(),   # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
            # You can add more data augmentation here for training
        ])
    
    
    #cifar_name = "CIFAR100"
    #cifar100 = load_dataset(dataset_name=cifar_name, transform=cifar_transform, dataset_path=dataset_path)
    cifar100 = DATASETS["cifar100"](transform=cifar_transform)
    cifar100_validation = DATASETS[f"cifar100-val"](transform=cifar_transform)

    caltech_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    

    caltech101_raw = DATASETS["caltech101"](transform=caltech_transform)
    #import deeplake
    #caltech101_raw = deeplake.load('hub://activeloop/caltech101')
    


    train_idx, val_idx = train_test_split(
        range(len(caltech101_raw)),
        test_size=0.2,  # 20% for validation
        random_state=0
    )

    # Create subsets for train and validation
    #caltech101 = torch.utils.data.Subset(caltech101_raw, train_idx)
    #caltech101_validation = torch.utils.data.Subset(caltech101_raw, val_idx)

    # display some images
    #for i in range(5):  # Check the first 5 images
    #    img, _ = caltech101[i]


    caltechtrain, caltechtest, caltechfull = get_caltech_loaders(num_workers=args.num_workers)


    run_counter = 1
    for run_iter in all_successful_runs.iterrows():
        print("currently computing metrics for run number", run_counter)
        run_counter += 1
        run_id = run_iter[1]['run_id']
        run_name = run_iter[1]['tags.mlflow.runName']
        model_uri = f"runs:/{run_id}/pretraining_model/"
        run = client.get_run(run_id)
        LOG.info(f"Checking run {run_name}")

        model = mlflow.pytorch.load_model(model_uri)
        

        if not ('CIFAR100_rank' in run.data.metrics):
            LOG.info(f"No CIFAR100 rank found for run: {run_id} computing...")
            #model.eval()
            rank = rank_me(model, cifar100, device=args.device)

            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("CIFAR100_rank", rank)
                mlflow.end_run()
                LOG.info(f"CIFAR100 rank estimation: {rank} for run: {run_id}")
        
        if not ('caltech101_rank' in run.data.metrics):
            LOG.info(f"No caltech101 rank found for run: {run_id} computing...")
            #model.eval()
            rank = rank_me(model, caltechfull, device=args.device)
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("caltech101_rank", rank)
                mlflow.end_run()
                LOG.info(f"caltech101 rank estimation: {rank} for run: {run_id}")

        if not ('CIFAR100_accuracy' in run.data.metrics):
            LOG.info(f"No CIFAR100 accuracy found for run: {run_id}, finetuning over {args.finetune_epochs} epochs...")
            model.train()

            accuracy = finetune.finetune_pipeline(
                model,
                testset=cifar100_validation,
                trainset=cifar100,
                epochs=args.finetune_epochs,
                num_workers=args.num_workers,
                device=args.device,
                n_classes=100
            )
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("CIFAR100_accuracy", accuracy)
                mlflow.end_run()
                LOG.info(f"CIFAR100 accuracy: {accuracy} for run: {run_id}")

        if not ('caltech101_accuracy' in run.data.metrics):
            LOG.info(f"No Caltech101 accuracy found for run: {run_id}, finetuning over {args.finetune_epochs} epochs...")

            model.train()

            accuracy = finetune_caltech.finetune_pipeline(
                model,
                trainloader=caltechtrain,
                testloader=caltechtest,
                epochs=args.finetune_epochs,
                num_workers=args.num_workers,
                device=args.device,
                n_classes=101
            )
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("caltech101_accuracy", accuracy)
                mlflow.end_run()
                LOG.info(f"Caltech101 accuracy: {accuracy} for run: {run_id}")


if __name__ == "__main__":
    if "MLFLOW_TRACKING_URI" not in os.environ:
        raise EnvironmentError("Environment variable MLFLOW_TRACKING_URI not "
                               "set. Example: set it with\n"
                               "export MLFLOW_TRACKING_URI=\"http://127.0.0.1:8080\"")
    main()
