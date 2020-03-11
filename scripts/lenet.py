import fire
import torch
import numpy as np
from time import time
from torch import nn
from torch import optim
import os
from torchvision import datasets, transforms

def FF(layer_sizes):
    # Build a feed-forward network
    layers = []
    for idx, size in enumerate(layer_sizes[:-1]):
        layers.append(nn.Linear(size, layer_sizes[idx+1]))
        layers.append(nn.ReLU())

    return nn.Sequential(*layers, nn.LogSoftmax(dim=1))

def train_mnist(hidden_sizes, out_dir, data_loc, epochs:int =10, lr:float =0.01, momentum:float=0.5):

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

    # Layer details for the neural network
    if torch.cuda.is_available():
        print("Using CUDA")
        import os
        print(os.environ.get("CUDA_VISIBLE_DEVICES"))
        device = torch.device('cuda')
    else:
        print("Not Using CUDA")
        device = torch.device('cpu')
    model = FF([784] + hidden_sizes + [10]).to(device)
    print()
    print(model)

    criterion = nn.NLLLoss()
    images, labels = next(iter(trainloader))
    images, labels = images.to(device), labels.to(device)
    images = images.view(images.shape[0], -1)

    logps = model(images)
    loss = criterion(logps, labels)

    # Optimizers require the parameters to optimize and a learning rate
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    time0 = time()
    for e in range(epochs):
        running_loss = 0
        for images, labels in trainloader:
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
            images, labels = images.to(device), labels.to(device)

            # Training pass
            optimizer.zero_grad()

            output = model(images)
            loss = criterion(output, labels)

            #This is where the model learns by backpropagating
            loss.backward()

            #And optimizes its weights here
            optimizer.step()

            running_loss += loss.item()
        else:
            print("Epoch {} - Training loss: {}".format(e, running_loss/len(trainloader)))

    print("\nTraining Time (in minutes) =",(time()-time0)/60)

    # Move the model back to cpu for validation
    model = model.to('cpu')

    correct_count, all_count = 0, 0
    for images,labels in valloader:
        for i in range(len(labels)):
            img = images[i].view(1, 784)
            # Turn off gradients to speed up this part
            with torch.no_grad():
                logps = model(img)

            # Output of the network are log-probabilities, need to take exponential for probabilities
            ps = torch.exp(logps)
            probab = list(ps.numpy()[0])
            pred_label = probab.index(max(probab))
            true_label = labels.numpy()[i]
            if(true_label == pred_label):
                correct_count += 1
            all_count += 1

    print("Number Of Images Tested =", all_count)
    print("\nModel Accuracy =", (correct_count/all_count))

    torch.save(model.state_dict(), out_dir)

if __name__ == "__main__":
    fire.Fire(train_mnist)
