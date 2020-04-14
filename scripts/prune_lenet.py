from coreset import compress_fc_layer
import torch
import fire

def prune_lenet(model_dir, out_dir, sparsity: float, prune_type, beta:float):
    params = torch.load(model_dir)
    new_layer1, new_layer2, indices = compress_fc_layer(
        (params['0.weight'], params['0.bias']),
        (params['2.weight'], params['2.bias']),
        int((1 - sparsity) * 300),
        torch.nn.ReLU(),
        beta,
        'cpu',
        prune_type
    )
    print(f'Selected these indices from layer 0: {indices}')
    new_layer2, new_layer3, indices = compress_fc_layer(
        new_layer2,
        (params['4.weight'], params['4.bias']),
        int((1 - sparsity) * 100),
        torch.nn.ReLU(),
        beta,
        'cpu',
        prune_type
    )
    print(f'Selected these indices from layer 1: {indices}')

    params['0.weight'], params['0.bias'] = new_layer1
    params['2.weight'], params['2.bias'] = new_layer2
    params['4.weight'], params['4.bias'] = new_layer3

    torch.save(params, out_dir)

if __name__ == '__main__':
    fire.Fire(prune_lenet)
