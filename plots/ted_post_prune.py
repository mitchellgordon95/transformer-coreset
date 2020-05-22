import matplotlib.pyplot as plt
import subprocess
import pandas as pd
from io import StringIO

fig, axes = plt.subplots()

ducttape = subprocess.Popen(["ducttape", "main.tape", "-C", "main.tconf", "-p", "ted", "summary", "post_prune"], stdout=subprocess.PIPE)
tabular = subprocess.Popen(['tabular'], stdin=ducttape.stdout, stdout=subprocess.PIPE)
csv = subprocess.check_output(["grep", "-o", "^[^#]*"], stdin=tabular.stdout).decode('ascii')

table = pd.read_csv(StringIO(csv), sep="\s+")
topk = table[table["PruneType"] == "topk"]
uniform = table[table["PruneType"] == "uniform"]
L1 = table[table["PruneType"] == "L1"]
coreset = table[table["PruneType"] == "coreset"]
randmatmul = table[table["PruneType"] == "randmatmul"]

axes.set_xlabel("Sparsity")
axes.set_ylabel("Dev BLEU")

axes.plot(uniform["Sparsity"], uniform["post_prune"], label="Uniform")
axes.plot(L1["Sparsity"], L1["post_prune"], label="L1")
axes.plot(topk["Sparsity"], topk["post_prune"], label="L2")
axes.plot(coreset["Sparsity"], coreset["post_prune"], label="Coreset")
axes.plot(randmatmul["Sparsity"], randmatmul["post_prune"], label="Random Mat Mul")
axes.set_title("TED Talks German-English Translation Post Prune")
axes.legend()
fig.savefig(f'plots_out/sparsity_vs_acc_ted_post_prune.png')
