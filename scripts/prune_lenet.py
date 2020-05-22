from coreset import compress_fc_layer
import torch
import fire
import sys

def prune_lenet(model_dir, out_dir, sparsity: float, prune_type, beta:float):
    params = torch.load(model_dir)
    layer1_sample_size = int((1 - sparsity) * 300)
    new_layer1, new_layer2, indices = compress_fc_layer(
        (params['0.weight'], params['0.bias']),
        (params['2.weight'], params['2.bias']),
        layer1_sample_size,
        torch.nn.ReLU(),
        beta,
        'cpu',
        prune_type
    )
    print(f'Selected these indices from layer 0: {indices}', file=sys.stderr)
    layer2_sample_size = int((1 - sparsity) * 100)
    new_layer2, new_layer3, indices = compress_fc_layer(
        new_layer2,
        (params['4.weight'], params['4.bias']),
        layer2_sample_size,
        torch.nn.ReLU(),
        beta,
        'cpu',
        prune_type
    )
    print(f'Selected these indices from layer 1: {indices}', file=sys.stderr)

    params['0.weight'], params['0.bias'] = new_layer1
    params['2.weight'], params['2.bias'] = new_layer2
    params['4.weight'], params['4.bias'] = new_layer3

    torch.save(params, out_dir)

    print(layer1_sample_size)
    print(layer2_sample_size)

if __name__ == '__main__':
    fire.Fire(prune_lenet)
