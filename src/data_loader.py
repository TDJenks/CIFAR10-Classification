import torch
import torchvision
import torchvision.transforms as transforms

def get_cifar10_loaders(batch_size=128, num_workers=2):
    # Transforms for training
    # Random crops and flips to help with learning
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # Easier transforms for testing
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    # Initializing CIFAR-10 datasets with respective transforms
    
    trainset = torchvision.datasets.CIFAR10(
        root="./data", 
        train=True, 
        download=True, 
        transform=transform_train
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", 
        train=False, 
        download=True, 
        transform=transform_test
    )

    # Initializing DataLoaders
    
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )

    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )

    return trainloader, testloader