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
from src.models.encoder_projector import EncoderProjector
from torchvision.transforms import transforms
import pandas as pd
from src.utils.logging import init_logging
from src.utils.pytorch_device import get_device
from src.utils.evaluate import rank_me

from src.utils.load_dataset import load_dataset, DATASETS

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
    cifar100 = DATASETS[args.dataset](transform=cifar_transform)

    for run_iter in all_successful_runs.iterrows():
        run_id = run_iter[1]['run_id']
        model_uri = f"runs:/{run_id}/pretraining_model/"
        run = client.get_run(run_id)

        model = mlflow.pytorch.load_model(model_uri)

        if not ('rank' in run.data.metrics):
            LOG.info("No rank found for run: ", run_id, "computing...")
            model.eval()
            rank = rank_me(model, cifar100, device=args.device)
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("rank", rank)
                mlflow.end_run()
                LOG.info("rank estimation: ", rank, "for run: ", run_id)
        
        if not ('CIFAR100_accuracy' in run.data.metrics):
            LOG.info("No CIFAR100 accuracy found for run: ", run_id, ", finetuning over", args.finetune_epochs, " epochs...")
            model.train()

            accuracy = finetune.finetune_pipeline(model, testset=cifar100, trainset=cifar100, epochs=args.finetune_epochs)
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric("CIFAR100_accuracy", accuracy)
                mlflow.end_run()
                LOG.info("CIFAR100 accuracy: ", accuracy, "for run: ", run_id)

        if not ('iNaturalist_accuracy' in run.data.metrics):
            LOG.info("No iNaturalist accuracy found for run: ", run_id, ", finetuning over", args.finetune_epochs, " epochs...")
            LOG.info("jk, not implemented yet")


if __name__ == "__main__":
    os.environ["MLFLOW_TRACKING_URI"] = "http://127.0.0.1:8080"
    main()
