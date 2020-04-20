import torch
import numpy as np


def prune_attn_uniform(k_proj, q_proj, v_proj, out_proj, sample_size):
    heads, head_dim, embed_dim = k_proj.shape

    new_k_proj = torch.zeros((heads,sample_size,k_proj.shape[-1]))
    new_q_proj = torch.zeros((heads,sample_size,q_proj.shape[-1]))
    new_v_proj = torch.zeros((heads,sample_size,v_proj.shape[-1]))
    new_out_proj = torch.zeros((heads,sample_size,out_proj.shape[-1]))

    for head in range(heads):
        indices = np.random.choice(head_dim, size=sample_size, replace=False)

        new_k_proj[head] = k_proj[head,indices,:]
        new_q_proj[head] = q_proj[head,indices,:]
        new_v_proj[head] = v_proj[head,indices,:]
        new_out_proj[head] = out_proj[head,indices,:]
        print(f"Head {head}: sampling indices {indices}")

    return new_k_proj, new_q_proj, new_v_proj, new_out_proj
