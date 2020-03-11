import sys
import fire
from torchvision import datasets

def download_mnist(loc):
    datasets.MNIST(loc, download=True, train=True)
    datasets.MNIST(loc, download=True, train=False)


if __name__ == "__main__":
    fire.Fire(download_mnist)
