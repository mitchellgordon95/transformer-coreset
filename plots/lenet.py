import matplotlib.pyplot as plt
from common import avg_trials

BETA = 45
fig, axes = plt.subplots()

# One {} for the first format, two {{}} for the second format
fname = 'out/agg_eval_FF/BetaFF.{BETA}+FF_lr.0.1+PruneTypeFF.{prune_type}+TestMode.no+TrialFF.{{trial}}/sparsity_vs_acc'

axes.set_xlabel("Sparsity")
axes.set_ylabel("Dev Acc")
axes.plot(*avg_trials(fname=fname.format(BETA=BETA, prune_type='Coreset'), trials=3), label="Coreset")
axes.plot(*avg_trials(fname=fname.format(BETA=BETA, prune_type='Uniform'), trials=3), label="Uniform")
axes.plot(*avg_trials(fname=fname.format(BETA=BETA, prune_type='Top-K'), trials=3), label="Top-K")
axes.set_title("LeNet-300-100 Pruning")
axes.legend()
fig.savefig(f'plots_out/sparsity_vs_acc_beta_{BETA}.png')
