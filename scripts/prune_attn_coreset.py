import torch
import numpy as np
import sys


def prune_attn_coreset(k_proj, q_proj, v_proj, out_proj, sample_size):
    heads, head_dim, embed_dim = k_proj.shape

    new_k_proj = torch.zeros((heads, sample_size, k_proj.shape[-1]))
    new_q_proj = torch.zeros((heads, sample_size, q_proj.shape[-1]))
    new_v_proj = torch.zeros((heads, sample_size, v_proj.shape[-1]))
    new_out_proj = torch.zeros((heads, sample_size, out_proj.shape[-1]))

    for head in range(heads):
        new_k_proj[head], new_q_proj[head], indices = sample_rows(k_proj[head], q_proj[head], sample_size)
        print(f"Head {head}: sampling indices {indices}, for QK", file=sys.stderr)

        new_v_proj[head], new_out_proj[head], indices = sample_rows(v_proj[head], out_proj[head], sample_size)
        print(f"Head {head}: sampling indices {indices}, for VO", file=sys.stderr)

    return new_k_proj, new_q_proj, new_v_proj, new_out_proj

def sample_rows(matrix1, matrix2, sample_size):
    """Samples rows according to the coreset probability distribution. Re-weights the second matrix accordingly."""
    assert matrix1.shape[0] == matrix2.shape[0]
    norms1 = torch.norm(matrix1, dim=1)
    norms2 = torch.max(matrix2, dim=1)[0]
    unnormalized_probs = norms1 * norms2
    probs = unnormalized_probs.numpy() / np.sum(unnormalized_probs.numpy())
    matrix2 = matrix2 / (sample_size * probs).reshape(-1,1)
    indices = np.random.choice(matrix1.shape[0], size=sample_size, p=probs, replace=True)
    return matrix1[indices], matrix2[indices], indices

