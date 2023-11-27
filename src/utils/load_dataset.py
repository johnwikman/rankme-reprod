import torchvision.transforms as transforms
import torchvision.datasets as datasets


def load_dataset(dataset_name, transform, dataset_path="_datasets/"):
    """
    Load a dataset for the RankMe pipeline.
    If the dataset is not found in the given path, it is downloaded (not for iNaturalist). We are using the "train" split of the dataset options are provided.

    Parameters:
    dataset_path (str): The path to the dataset.
    dataset_name (str): The name of the dataset, e.g 'iNaturalist' or 'CIFAR100'.
    transform (torchvision.transforms): The transforms to apply to the dataset.

    Returns:
    torch.utils.data.Dataset: The loaded PyTorch dataset.
    """

    dataset_name = dataset_name.lower()

    try:
        if dataset_name == "cifar10":
            # Load or download CIFAR100
            dataset = datasets.CIFAR10(
                root=dataset_path, train=True, download=True, transform=transform
            )

        elif dataset_name == "cifar100":
            # Load or download CIFAR100
            dataset = datasets.CIFAR100(
                root=dataset_path, train=True, download=True, transform=transform
            )

        elif dataset_name == "inaturalist":
            # Load or download iNaturalist
            # You can customize 'version' and 'target_type' as per your requirement
            dataset = datasets.INaturalist(
                root=dataset_path,
                version="2021_train_mini",
                target_type="full",
                transform=transform,
                download=False,  # tänker inte ladda ner 224gb >:(
            )

        elif dataset_name == "imagenet":
            # Load or download ImageNet
            # You can customize 'split' as per your requirement
            dataset = datasets.ImageFolder(
                root=dataset_path + "tiny-imagenet-200/train/",
                transform=transform,
            )

        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        return dataset
    # except missing folder
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Please download the dataset and try again."
        )
