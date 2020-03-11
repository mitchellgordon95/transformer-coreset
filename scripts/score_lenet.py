import fire
import torch
from torchvision import datasets, transforms
from lenet import FF

def score_lenet(model_dir, data_loc):

    # Define a transform to normalize the data
    # Mean and std from pytorch example
    # https://github.com/pytorch/examples/blob/master/mnist/main.py
    transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,)),
                                ])

    # Download and load the training data
    valset = datasets.MNIST(data_loc, download=False, train=False, transform=transform)
    valloader = torch.utils.data.DataLoader(valset, batch_size=64, shuffle=True)

    params = torch.load(model_dir)
    hidden_sizes = [params['0.weight'].shape[0], params['2.weight'].shape[0]]
    print('Found a model with layer sizes {hidden_sizes}')
    model = FF([784] + hidden_sizes + [10])
    model.load_state_dict(params)

    correct_count, all_count = 0, 0
    for images, labels in valloader:
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


if __name__ == "__main__":
    fire.Fire(score_lenet)
