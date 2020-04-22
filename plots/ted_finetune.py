import matplotlib.pyplot as plt
from common import avg_trials

fig, axes = plt.subplots()

# One {} for the first format, two {{}} for the second format
fname = 'out/agg_bleu_post_finetune/Dataset.ted+Lang.deen+ModelSize.large+PruneType.{prune_type}+PruneTypeMLP.none+Trial.{{trial}}/sparsity_vs_bleu'

axes.set_xlabel("Sparsity")
axes.set_ylabel("Dev BLEU")
axes.plot(*avg_trials(fname.format(prune_type='uniform'), 1), label="Uniform")
axes.plot(*avg_trials(fname.format(prune_type='topk'), 1), label="TopK")
axes.set_title("TED Talks German-English Translation")
axes.legend()
fig.savefig(f'plots_out/sparsity_vs_acc_ted_finetune.png')
