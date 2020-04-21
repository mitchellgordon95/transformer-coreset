import matplotlib.pyplot as plt
import numpy as np
import os
import torch

fig, axes = plt.subplots()

vals = []
dirname = 'out/attention_values/Dataset.ted/tmp'
count = 0
for fname in os.listdir(dirname):
    print(count)
    count += 1
    fname = os.path.join(dirname, fname)
    array = torch.load(fname, map_location=torch.device('cpu')).numpy()
    array = array[np.isfinite(array)].flatten()
    vals.append(array)

axes.set_xlabel("Buckets")
axes.set_ylabel("Freq")
axes.hist(np.concatenate(vals), bins=50, range=(-5,5))
axes.set_title("Pre-softmax Attn Activations")
fig.savefig(f'plots_out/pre_softmax.png')
