import "tapes/submitters.tape"
import "tapes/versioners.tape"

## Tasks start here
import "tapes/data_prep.tape"
import "tapes/train.tape"
import "tapes/pruning.tape"
import "tapes/finetuning.tape"

# FF experiments (MNIST, VGG, etc.)
import "tapes/FF.tape"

plan all {
  reach bleu_dev via (Dataset: ted) * (Lang: deen)
}

plan FF {
  reach eval_FF via (SparsityFF: *) * (PruneTypeFF: *)
}

# Nuts and bolts:
global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}

