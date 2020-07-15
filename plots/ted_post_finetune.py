import matplotlib.pyplot as plt
import subprocess
import pandas as pd
from io import StringIO


ducttape = subprocess.Popen(["ducttape", "main.tape", "-C", "main.tconf", "-p", "ted", "summary", "post_finetune"], stdout=subprocess.PIPE)
tabular = subprocess.Popen(['tabular'], stdin=ducttape.stdout, stdout=subprocess.PIPE)
csv = subprocess.check_output(["grep", "-o", "^[^#]*"], stdin=tabular.stdout).decode('ascii')

table = pd.read_csv(StringIO(csv), sep="\s+")
table['Sparsity'] = pd.to_numeric(table['Sparsity'], errors='raise')
table['post_finetune'] = pd.to_numeric(table['post_finetune'], errors='coerce')

fig, axes = plt.subplots()

axes.set_xlabel("Sparsity")
axes.set_ylabel("Dev BLEU")

for prune_type in ["L2", "L1"]:
    rel = table[(table["PruneType"] == prune_type)]
    axes.plot(rel["Sparsity"], rel["post_finetune"], label=prune_type)

for prune_type in ["uniform", "coreset", "randmatmul"]:
    for replacement, scaling in [(False, True)]: #[(True, False), (False, True), (False, False)]:
        rel = table[(table["PruneType"] == prune_type) & (table["WithScaling"] == scaling) & (table["WithReplacement"] == replacement)]
        label = prune_type
        # if replacement:
        #     label += "_replace"
        # if scaling:
        #     label += "_scaling"
        axes.plot(rel["Sparsity"], rel["post_finetune"], label=label)

axes.set_title("TED Talks German-English Translation Post Finetune")
axes.legend()
fig.savefig(f'plots_out/sparsity_vs_acc_ted_post_finetune.png')
