import torch
import torchvision
import torchvision.transforms as transforms

# src/train/train.py

from data_loader import get_cifar10_loaders

def main():
    trainloader, testloader = get_cifar10_loaders(batch_size=128)

    # Example: sanity check
    print("Training batches:", len(trainloader))
    print("Testing batches:", len(testloader))

    # TODO: pass these loaders into your training loop

if __name__ == "__main__":
    main()