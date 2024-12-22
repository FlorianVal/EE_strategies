import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import logging

logger = logging.getLogger(__name__)

# Define constants
DATA_DIR = "data"
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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


def get_dataset(name, train=True):
    if name == "cifar10":
        return get_cifar10(train)
    elif name == "cifar100":
        return get_cifar100(train)
    elif name == "imagenet_subset":
        return get_imagenet_subset(train)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def create_data_loaders(
    dataset_name: str, batch_size: int, num_workers: int = 2, seed: int = 42
) -> tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and testing.
    
    Args:
        dataset_name (str): Name of the dataset ('cifar10' or 'cifar100')
        batch_size (int): Batch size for DataLoader
        num_workers (int): Number of workers for data loading
        seed (int): Random seed for reproducibility
    
    Returns:
        tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    # Get the training and test datasets
    train_dataset = get_dataset(dataset_name, train=True)
    test_dataset = get_dataset(dataset_name, train=False)

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

    return train_loader, test_loader


# Example usage
if __name__ == "__main__":
    train_loader, test_loader = create_data_loaders(
        "cifar100", batch_size=64, num_workers=4
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
