import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def benchmark_workers(workers_list, batch_size=128, num_batches=100):
    transform = transforms.ToTensor()
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    print(f"Benchmarking with batch_size={batch_size}, num_batches={num_batches}\n")

    for workers in workers_list:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers
        )

        start = time.time()
        for i, (x, y) in enumerate(loader):
            if i >= num_batches:
                break
        end = time.time()

        print(f"num_workers={workers:<2} | Time for {num_batches} batches: {end - start:.2f} sec")

if __name__ == "__main__":
    # Try different worker settings
    workers_to_test = [0, 1, 2, 4, 8]
    benchmark_workers(workers_to_test)