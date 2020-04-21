import matplotlib.pyplot as plt
from common import avg_trials

fig, axes = plt.subplots()

# One {} for the first format, two {{}} for the second format
fname = 'out/agg_bleu_post_prune/Dataset.ted+Lang.deen+ModelSize.large+PruneType.{prune_type}+Trial.{{trial}}/sparsity_vs_bleu'

uniform_sparsities, uniform_accs = 

axes.set_xlabel("Sparsity")
axes.set_ylabel("Dev BLEU")
axes.plot(*avg_trials(fname.format(prune_type='uniform'), 1), label="Uniform")
axes.set_title("TED Talks German-English Translation")
axes.legend()
fig.savefig(f'plots_out/sparsity_vs_acc_ted.png')
