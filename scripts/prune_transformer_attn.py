import numpy as np
import torch
import fire
from prune_attn_uniform import prune_attn_uniform
from prune_attn_topk import prune_attn_topk
import sys


def prune_transformer_attn(model_chkpt, out_dir, sparsity: float, method):
    chkpt = torch.load(model_chkpt, map_location=torch.device('cpu'))

    encoder_dim = chkpt['args'].encoder_embed_dim
    decoder_dim = chkpt['args'].decoder_embed_dim

    encoder_heads = chkpt['args'].encoder_attention_heads
    decoder_heads = chkpt['args'].decoder_attention_heads

    encoder_head_dim = encoder_dim // encoder_heads
    decoder_head_dim = decoder_dim // decoder_heads

    encoder_sample_size = int((1 - sparsity) * encoder_head_dim)
    decoder_sample_size = int((1 - sparsity) * decoder_head_dim)

    for layer in range(6):
        for half in ['encoder', 'decoder']:
            print(f'layer {layer} {half}', file=sys.stderr)
            if half == 'encoder':
                prune_MHA(chkpt, f'{half}.layers.{layer}.self_attn', encoder_heads, encoder_head_dim, encoder_sample_size, method)
            if half == 'decoder':
                prune_MHA(chkpt, f'{half}.layers.{layer}.encoder_attn', decoder_heads, decoder_head_dim, decoder_sample_size, method)
                prune_MHA(chkpt, f'{half}.layers.{layer}.self_attn', decoder_heads, decoder_head_dim, decoder_sample_size, method)

    # Set the internal attention projection dimension
    chkpt['args'].encoder_attn_proj_dim = encoder_sample_size * encoder_heads
    chkpt['args'].decoder_attn_proj_dim = decoder_sample_size * decoder_heads
    torch.save(chkpt, out_dir)

    print(encoder_sample_size * encoder_heads)
    print(decoder_sample_size * decoder_heads)


def prune_MHA(chkpt, prefix, attention_heads, head_dim, sample_size, method):
    # Remember, every row of a projection matrix corresponds to a dimension of the key/query/value
    # Furthermore, these rows are divided evenly among the projection heads.
    k_proj = torch.cat([
        chkpt['model'][f'{prefix}.k_proj.weight'].cpu(),
        chkpt['model'][f'{prefix}.k_proj.bias'].cpu().reshape((-1,1))], dim=1)
    q_proj = torch.cat([
        chkpt['model'][f'{prefix}.q_proj.weight'].cpu(),
        chkpt['model'][f'{prefix}.q_proj.bias'].cpu().reshape((-1,1))], dim=1)
    v_proj = torch.cat([
        chkpt['model'][f'{prefix}.v_proj.weight'].cpu(),
        chkpt['model'][f'{prefix}.v_proj.bias'].cpu().reshape((-1,1))], dim=1)
    # Note: we don't have to prune the bias of the output projection, since a "neuron" is an output dimension
    out_proj_T = chkpt['model'][f'{prefix}.out_proj.weight'].T

    k_proj = k_proj.reshape((attention_heads, head_dim, k_proj.shape[-1]))
    q_proj = q_proj.reshape((attention_heads, head_dim, q_proj.shape[-1]))
    v_proj = v_proj.reshape((attention_heads, head_dim, v_proj.shape[-1]))
    out_proj_T = out_proj_T.reshape((attention_heads, head_dim, out_proj_T.shape[-1]))

    # Don't forget to think about how these pair up, and how we might exploit that to speed up W_Q W_K
    if method == 'uniform':
        k_proj, q_proj, v_proj, out_proj_T = prune_attn_uniform(k_proj, q_proj, v_proj, out_proj_T, sample_size)
    if method == 'topk':
        k_proj, q_proj, v_proj, out_proj_T = prune_attn_topk(k_proj, q_proj, v_proj, out_proj_T, sample_size)
    else:
        raise Exception("Unknown pruning type.")

    new_embed_dim = sample_size * attention_heads
    k_proj = k_proj.reshape((new_embed_dim, k_proj.shape[-1]))
    q_proj = q_proj.reshape((new_embed_dim, q_proj.shape[-1]))
    v_proj = v_proj.reshape((new_embed_dim, v_proj.shape[-1]))
    out_proj_T = out_proj_T.reshape((new_embed_dim, out_proj_T.shape[-1]))

    chkpt['model'][f'{prefix}.k_proj.weight'], chkpt['model'][f'{prefix}.k_proj.bias'] = k_proj[:,:-1], k_proj[:,-1].reshape(-1)
    chkpt['model'][f'{prefix}.q_proj.weight'], chkpt['model'][f'{prefix}.q_proj.bias'] = q_proj[:,:-1], q_proj[:,-1].reshape(-1)
    chkpt['model'][f'{prefix}.v_proj.weight'], chkpt['model'][f'{prefix}.v_proj.bias'] = v_proj[:,:-1], v_proj[:,-1].reshape(-1)
    chkpt['model'][f'{prefix}.out_proj.weight'] = out_proj_T.T

if __name__ == '__main__':
    fire.Fire(prune_transformer_attn)
