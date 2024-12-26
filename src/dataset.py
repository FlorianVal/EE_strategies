import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Define constants
DATA_DIR = "data"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Dataset information
DATASET_INFO = {
    "cifar10": {"num_classes": 10, "mean": CIFAR10_MEAN, "std": CIFAR10_STD},
    "cifar100": {"num_classes": 100, "mean": CIFAR100_MEAN, "std": CIFAR100_STD},
    "imagenet_subset": {"num_classes": 1000, "mean": IMAGENET_MEAN, "std": IMAGENET_STD}
}

def get_cifar10(train=True):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)]
    )
    return datasets.CIFAR10(
        root=DATA_DIR, train=train, download=True, transform=transform
    )

def get_cifar100(train=True):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)]
    )
    return datasets.CIFAR100(
        root=DATA_DIR, train=train, download=True, transform=transform
    )

def get_imagenet_subset(train=True):
    raise NotImplementedError("ImageNet subset is not implemented yet.")

def get_dataset(name: str, train: bool = True):
    """Get dataset and its number of classes."""
    if name not in DATASET_INFO:
        raise ValueError(f"Unknown dataset: {name}")
    
    if name == "cifar10":
        dataset = get_cifar10(train)
    elif name == "cifar100":
        dataset = get_cifar100(train)
    elif name == "imagenet_subset":
        dataset = get_imagenet_subset(train)
    
    return dataset, DATASET_INFO[name]["num_classes"]

def create_data_loaders(
    dataset_name: str, batch_size: int, num_workers: int = 2, seed: int = 42
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create data loaders for training and testing.
    
    Args:
        dataset_name (str): Name of the dataset ('cifar10' or 'cifar100')
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for data loading
        seed (int): Random seed for reproducibility
    
    Returns:
        Tuple[DataLoader, DataLoader, int]: (train_loader, test_loader, num_classes)
    """
    # Get the training and test datasets
    train_dataset, num_classes = get_dataset(dataset_name, train=True)
    test_dataset, _ = get_dataset(dataset_name, train=False)

    logger.info(f"Dataset: {dataset_name} (num_classes: {num_classes})")
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, test_loader, num_classes

# Example usage
if __name__ == "__main__":
    train_loader, test_loader, num_classes = create_data_loaders(
        "cifar100", batch_size=64, num_workers=4
    )
    print(f"Number of classes: {num_classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
