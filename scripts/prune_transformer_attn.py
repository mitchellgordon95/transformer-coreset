import numpy as np
import torch
import fire
from prune_attn_uniform import prune_attn_uniform


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

    for half in ['encoder', 'decoder']:
        if half == 'encoder':
            embed_dim = encoder_dim
            head_dim = encoder_head_dim
            sample_size = encoder_sample_size
            attention_heads = encoder_heads
        if half == 'decoder':
            embed_dim = decoder_dim
            head_dim = decoder_head_dim
            sample_size = decoder_sample_size
            attention_heads = decoder_heads

        for layer in range(6):
            print(f'{half} layer {layer}')
            # Remember, every row of a projection matrix corresponds to a dimension of the key/query/value
            # Furthermore, these rows are divided evenly among the projection heads.
            k_proj = np.concatenate([
                chkpt['model'][f'{half}.layers.{layer}.self_attn.k_proj.weight'].cpu().numpy(),
                chkpt['model'][f'{half}.layers.{layer}.self_attn.k_proj.bias'].cpu().numpy().reshape((-1,1))], axis=1)
            q_proj = np.concatenate([
                chkpt['model'][f'{half}.layers.{layer}.self_attn.q_proj.weight'].cpu().numpy(),
                chkpt['model'][f'{half}.layers.{layer}.self_attn.q_proj.bias'].cpu().numpy().reshape((-1,1))], axis=1)
            v_proj = np.concatenate([
                chkpt['model'][f'{half}.layers.{layer}.self_attn.v_proj.weight'].cpu().numpy(),
                chkpt['model'][f'{half}.layers.{layer}.self_attn.v_proj.bias'].cpu().numpy().reshape((-1,1))], axis=1)
            out_proj = np.concatenate([
                chkpt['model'][f'{half}.layers.{layer}.self_attn.out_proj.weight'].cpu().numpy(),
                chkpt['model'][f'{half}.layers.{layer}.self_attn.out_proj.bias'].cpu().numpy().reshape((-1,1))], axis=1)

            k_proj = k_proj.reshape((attention_heads, head_dim, k_proj.shape[-1]))
            q_proj = q_proj.reshape((attention_heads, head_dim, q_proj.shape[-1]))
            v_proj = v_proj.reshape((attention_heads, head_dim, v_proj.shape[-1]))
            out_proj = out_proj.reshape((attention_heads, head_dim, out_proj.shape[-1]))

            # Don't forget to think about how these pair up, and how we might exploit that to speed up W_Q W_K
            if method == 'uniform':
                k_proj, q_proj, v_proj, out_proj = prune_attn_uniform(k_proj, q_proj, v_proj, out_proj, sample_size)
            else:
                raise Exception("Unknown pruning type.")

            new_embed_dim = sample_size * attention_heads
            k_proj = k_proj.reshape((new_embed_dim, k_proj.shape[-1]))
            q_proj = q_proj.reshape((new_embed_dim, q_proj.shape[-1]))
            v_proj = v_proj.reshape((new_embed_dim, v_proj.shape[-1]))
            out_proj = out_proj.reshape((new_embed_dim, out_proj.shape[-1]))

            chkpt['model'][f'{half}.layers.{layer}.self_attn.k_proj.weight'], chkpt['model'][f'{half}.layers.{layer}.self_attn.k_proj.bias'] = k_proj[:,:-1], k_proj[:,-1].ravel()
            chkpt['model'][f'{half}.layers.{layer}.self_attn.q_proj.weight'], chkpt['model'][f'{half}.layers.{layer}.self_attn.q_proj.bias'] = q_proj[:,:-1], q_proj[:,-1].ravel()
            chkpt['model'][f'{half}.layers.{layer}.self_attn.v_proj.weight'], chkpt['model'][f'{half}.layers.{layer}.self_attn.v_proj.bias'] = v_proj[:,:-1], v_proj[:,-1].ravel()
            chkpt['model'][f'{half}.layers.{layer}.self_attn.out_proj.weight'], chkpt['model'][f'{half}.layers.{layer}.self_attn.out_proj.bias'] = out_proj[:,:-1], out_proj[:,-1].ravel()


    # Set the internal attention projection dimension
    chkpt['args'].encoder_attn_proj_dim = encoder_sample_size * encoder_heads
    chkpt['args'].decoder_attn_proj_dim = decoder_sample_size * decoder_heads
    torch.save(chkpt, out_dir)

if __name__ == '__main__':
    fire.Fire(prune_transformer_attn)
