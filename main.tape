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
  reach agg_bleu_post_prune via (Dataset: ted) * (Lang: deen) * (Trial: 1) * (Sparsity: *) * (PruneType: uniform topk) * (PruneTypeMLP: none) 
  reach agg_bleu_post_finetune via (Dataset: ted) * (Lang: deen) * (Trial: 1) * (Sparsity: *) * (PruneType: uniform topk) * (PruneTypeMLP: none) * (ReinitAndTrain: *) 
}

plan from_scratch {
  # Take the uniform pruned models, throw out the weights, reinitialize and train to completion
  reach agg_bleu_post_finetune via (Dataset: ted) * (Lang: deen) * (Trial: 1) * (Sparsity: *) * (PruneType: uniform) * (PruneTypeMLP: none) * (ReinitAndTrain: yes) 
}

plan attention_values {
  reach attention_values via (Dataset: ted) * (Lang: deen)
}

plan FF {
  reach agg_eval_FF via (SparsityFF: *) * (PruneTypeFF: *) * (TrialFF: 3) * (BetaFF: *)
  reach agg_eval_post_finetune_FF via (SparsityFF: *) * (PruneTypeFF: *) * (TrialFF: 1) * (BetaFF: 45)
}

# Nuts and bolts:
global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}

