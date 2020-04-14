import fire
import torch
from torchvision import datasets, transforms
from lenet import FF
import matplotlib.pyplot as plt

def lenet_stats(model_dir, data_loc):

    # Define a transform to normalize the data
    # Mean and std from pytorch example
    # https://github.com/pytorch/examples/blob/master/mnist/main.py
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                                ])

    # Download and load the training data
    trainset = datasets.MNIST(data_loc, download=False, train=True, transform=transform)
    valset = datasets.MNIST(data_loc, download=False, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    model = FF([784, 300, 100, 10])
    model.load_state_dict(torch.load(model_dir))

    norms = []
    first_layer_norms = []
    for images, labels in trainloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            norms.append(torch.norm(img).item())

            preact = model[0](img)
            postact = model[1](img)
            first_layer_norms.append(torch.norm(postact).item())

    fig, axes = plt.subplots()
    axes.hist(norms)
    axes.set_title("MNIST Input Norms")
    fig.savefig('plots_out/mnist_input_norms.png')

    fig, axes = plt.subplots()
    axes.hist(first_layer_norms)
    axes.set_title("LeNet First Hidden Layer Norms")
    fig.savefig('plots_out/lenet_hidden_300_norms.png')


if __name__ == "__main__":
    fire.Fire(lenet_stats)
