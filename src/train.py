import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

from data_loader import get_cifar10_loaders

from model import CIFAR10CNN

def main():
    trainloader, testloader = get_cifar10_loaders(batch_size=128)

    # sanity check
    print("Training batches:", len(trainloader))
    print("Testing batches:", len(testloader))

    model = CIFAR10CNN()

    # Define loss & optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # 3. Training loop (to be implemented)

    # TODO: pass loaders into training loop

if __name__ == "__main__":
    main()