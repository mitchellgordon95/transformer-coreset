import csv
import numpy as np

def read_csv(fname):
    items = []
    with open(fname) as csv_f:
        for pair in csv.reader(csv_f, delimiter="\t"):
            items.append((float(pair[0]), float(pair[1])))
    items = sorted(items)
    return [item[0] for item in items], [item[1] for item in items]

def avg_trials(fname, trials):
    acc_list = []
    for trial in range(1,trials+1):
        sparsities, accs = read_csv(fname.format(trial=trial))
        acc_list.append(np.array(accs))

    avg = np.mean(np.stack(acc_list), axis=0)
    return sparsities, avg
