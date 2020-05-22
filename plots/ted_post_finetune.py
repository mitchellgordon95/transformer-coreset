import matplotlib.pyplot as plt
import subprocess
import pandas as pd
from io import StringIO

fig, axes = plt.subplots()

ducttape = subprocess.Popen(["ducttape", "main.tape", "-C", "main.tconf", "-p", "ted", "summary", "post_finetune"], stdout=subprocess.PIPE)
tabular = subprocess.Popen(['tabular'], stdin=ducttape.stdout, stdout=subprocess.PIPE)
csv = subprocess.check_output(["grep", "-o", "^[^#]*"], stdin=tabular.stdout).decode('ascii')

table = pd.read_csv(StringIO(csv), sep="\s+")
topk = table[table["PruneType"] == "topk"]
uniform = table[table["PruneType"] == "uniform"]

axes.set_xlabel("Sparsity")
axes.set_ylabel("Dev BLEU")

axes.plot(uniform["Sparsity"], uniform["post_finetune"], label="Uniform")
axes.plot(topk["Sparsity"], topk["post_finetune"], label="Top-K")
axes.set_title("TED Talks German-English Translation Post Finetune")
axes.legend()
fig.savefig(f'plots_out/sparsity_vs_acc_ted_post_finetune.png')
