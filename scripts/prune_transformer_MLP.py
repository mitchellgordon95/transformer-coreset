import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Add the parent directory to the python PATH
from coreset import compress_fc_layer
import torch
import fire

def prune_transformer_MLP_uniform(model_chkpt, out_dir, sparsity: float):
    chkpt = torch.load(model_chkpt, map_location=torch.device('cpu'))

    encoder_FF_size = chkpt['args'].encoder_ffn_embed_dim
    decoder_FF_size = chkpt['args'].decoder_ffn_embed_dim
    encoder_sample_size = int((1 - sparsity) * encoder_FF_size)
    decoder_sample_size = int((1 - sparsity) * decoder_FF_size)

    for half in ['encoder', 'decoder']:
        if half == 'encoder':
            sample_size, num_neurons = encoder_sample_size, encoder_FF_size
        if half == 'decoder':
            sample_size, num_neurons = decoder_sample_size, decoder_FF_size

        for layer in range(6):
            W1, bias1 = chkpt['model'][f'{half}.layers.{layer}.fc1.weight'], chkpt['model'][f'{half}.layers.{layer}.fc1.bias']
            W2, bias2 = chkpt['model'][f'{half}.layers.{layer}.fc2.weight'], chkpt['model'][f'{half}.layers.{layer}.fc2.bias']
            assert W1.shape[0] == W2.shape[1] == num_neurons  # Neurons are rows of W1, columns of W2
            new_layer1, new_layer2, indices = compress_fc_layer(
                (W1, bias1),
                (W2, bias2),
                sample_size,
                torch.nn.ReLU(),
                0, # Unused
                'cpu',
                'Uniform'
            )
            print(f'Selected these indices for layer {layer}: {indices}', file=sys.stderr)

            chkpt['model'][f'{half}.layers.{layer}.fc1.weight'], chkpt['model'][f'{half}.layers.{layer}.fc1.bias'] = new_layer1
            chkpt['model'][f'{half}.layers.{layer}.fc2.weight'], chkpt['model'][f'{half}.layers.{layer}.fc2.bias'] = new_layer2

    chkpt['args'].encoder_ffn_embed_dim = encoder_sample_size
    chkpt['args'].decoder_ffn_embed_dim = decoder_sample_size

    torch.save(chkpt, out_dir)
    print(encoder_sample_size)
    print(decoder_sample_size)

if __name__ == '__main__':
    fire.Fire(prune_transformer_MLP_uniform)
