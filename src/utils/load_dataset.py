import torchvision.transforms as transforms
import torchvision.datasets as datasets

DATASETS = {
    "cifar10": lambda dataset_dir, transform, train=True, download=True: \
        datasets.CIFAR10(
            dataset_dir,
            train=train,
            transform=transform,
            download=download,
        ),
}

def load_dataset(dataset_path, dataset_name):
    """
    Load an out-of-distribution dataset, either 'iNaturalist' or 'CIFAR100'.
    If the dataset is not found in the given path, it is downloaded.

    Parameters:
    dataset_path (str): The path to the dataset.
    dataset_name (str): The name of the dataset, either 'iNaturalist' or 'CIFAR100'.

    Returns:
    torch.utils.data.Dataset: The loaded PyTorch dataset.
    """

    # Define common transforms; these can be customized as needed
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if dataset_name == 'CIFAR100':
        # Load or download CIFAR100
        dataset = datasets.CIFAR100(root=dataset_path, train=True, download=True, transform=transform)

    elif dataset_name == 'iNaturalist':
        # Load or download iNaturalist
        # You can customize 'version' and 'target_type' as per your requirement
        dataset = datasets.INaturalist(root=dataset_path, version='2021_train_mini', target_type='full', transform=transform, download=True)

    elif dataset_name == 'ImageNet':
        # Load or download ImageNet
        # You can customize 'split' as per your requirement
        dataset = datasets.ImageNet(root=dataset_path, split='train', transform=transform)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return dataset