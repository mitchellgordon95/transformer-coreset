import torch
import numpy as np
import sys


def prune_attn_topk(k_proj, q_proj, v_proj, out_proj, sample_size):
    heads, head_dim, embed_dim = k_proj.shape

    new_k_proj = torch.zeros((heads, sample_size, k_proj.shape[-1]))
    new_q_proj = torch.zeros((heads, sample_size, q_proj.shape[-1]))
    new_v_proj = torch.zeros((heads, sample_size, v_proj.shape[-1]))
    new_out_proj = torch.zeros((heads, sample_size, out_proj.shape[-1]))

    for head in range(heads):
        new_k_proj[head], new_q_proj[head], indices = topk_rows(k_proj[head], q_proj[head], sample_size)
        print(f"Head {head}: sampling indices {indices}, for QK", file=sys.stderr)

        new_v_proj[head], new_out_proj[head], indices = topk_rows(v_proj[head], out_proj[head], sample_size)
        print(f"Head {head}: sampling indices {indices}, for VO", file=sys.stderr)

    return new_k_proj, new_q_proj, new_v_proj, new_out_proj

def topk_rows(matrix1, matrix2, sample_size):
    """Concatenates the matrices along dim 1, selects the rows with the highest norm of the concat."""
    combined = torch.cat([matrix1, matrix2], dim=1)
    norms = torch.norm(combined, dim=1)
    topk_indices, _ = torch.sort(torch.argsort(norms, descending=True)[:sample_size])
    return matrix1[topk_indices], matrix2[topk_indices], topk_indices

