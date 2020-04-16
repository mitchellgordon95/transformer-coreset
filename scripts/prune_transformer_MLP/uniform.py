import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the parent directory to the python PATH
from coreset import compress_fc_layer
import torch
import fire

def prune_transformer_MLP_uniform(model_dir, out_dir, sparsity: float):
    chkpt = torch.load(model_dir, map_location=torch.device('cpu'))

    for half in ['encoder', 'decoder']:
        for layer in range(6):
            W1, bias1 = chkpt['model'][f'{half}.layers.{layer}.fc1.weight'], chkpt['model'][f'{half}.layers.{layer}.fc1.bias']
            W2, bias2 = chkpt['model'][f'{half}.layers.{layer}.fc2.weight'], chkpt['model'][f'{half}.layers.{layer}.fc2.bias']
            assert W1.shape[0] == W2.shape[1] # Neurons are rows of W1, columns of W2
            new_layer1, new_layer2, indices = compress_fc_layer(
                (W1, bias1),
                (W2, bias2),
                int((1 - sparsity) * W1.shape[0]),
                torch.nn.ReLU(),
                0, # Unused
                'cpu',
                'Uniform'
            )
            print(f'Selected these indices for layer {layer}: {indices}')

            chkpt['model'][f'{half}.layers.{layer}.fc1.weight'], chkpt['model'][f'{half}.layers.{layer}.fc1.bias'] = new_layer1
            chkpt['model'][f'{half}.layers.{layer}.fc2.weight'], chkpt['model'][f'{half}.layers.{layer}.fc2.bias'] = new_layer2

    torch.save(chkpt, out_dir)

if __name__ == '__main__':
    fire.Fire(prune_transformer_MLP_uniform)
