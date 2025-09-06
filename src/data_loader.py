import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_loaders(batch_size=128, num_workers=0, val_split=0.1):
    # Transforms for training
    # Random crops and flips to help with learning
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # Transforms for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # Full training set (to be split into train + val)
    full_trainset = torchvision.datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform_train
    )

    # True test set
    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform_test
    )

    # Split into train and val
    val_size = int(len(full_trainset) * val_split)
    train_size = len(full_trainset) - val_size
    train_subset, val_subset = torch.utils.data.random_split(full_trainset, [train_size, val_size])

    # No weird transforms for validation
    val_subset.dataset.transform = transform_test

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader