import "tapes/submitters.tape"
import "tapes/versioners.tape"

## Tasks start here
import "tapes/data_prep.tape"
import "tapes/train.tape"
import "tapes/pruning.tape"
import "tapes/finetuning.tape"

plan ted {
  reach bleu_dev_post_prune via (Dataset: ted) * (Lang: deen) * (ShareQK: yes) * (Trial: 1) * (Sparsity: 0 0.3 0.6 0.9) * (PruneType: *) * (WithScaling: True) * (PruneTypeMLP: none) 
  reach bleu_dev_post_finetune via (Dataset: ted) * (Lang: deen) * (ShareQK: yes) * (Trial: 1) * (Sparsity: 0 0.3 0.6 0.9) * (PruneType: *) * (WithScaling: True) * (PruneTypeMLP: none) * (ReinitAndTrain: no) 

  # Sampling-based methods with no scaling
  reach bleu_dev_post_prune via (Dataset: ted) * (Lang: deen) * (ShareQK: yes) * (Trial: 1) * (Sparsity: 0 0.3 0.6 0.9) * (PruneType: uniform coreset randmatmul) * (PruneTypeMLP: none) 
  reach bleu_dev_post_finetune via (Dataset: ted) * (Lang: deen) * (ShareQK: yes) * (Trial: 1) * (Sparsity: 0 0.3 0.6 0.9) * (PruneType: uniform coreset randmatmul) * (PruneTypeMLP: none) * (ReinitAndTrain: no) 

  # Sampling-based methods With replacement
  reach bleu_dev_post_prune via (Dataset: ted) * (Lang: deen) * (ShareQK: yes) * (Trial: 1) * (Sparsity: 0 0.3 0.6 0.9) * (PruneType: uniform coreset randmatmul) * (WithReplacement: True) * (PruneTypeMLP: none) 
  reach bleu_dev_post_finetune via (Dataset: ted) * (Lang: deen) * (ShareQK: yes) * (Trial: 1) * (Sparsity: 0 0.3 0.6 0.9) * (PruneType: uniform coreset randmatmul) * (WithReplacement: True) * (PruneTypeMLP: none) * (ReinitAndTrain: no) 
}

plan from_scratch {
  # Take the uniform pruned models, throw out the weights, reinitialize and train to completion
  reach agg_bleu_post_finetune via (Dataset: ted) * (Lang: deen) * (Trial: 1) * (Sparsity: *) * (PruneType: uniform) * (PruneTypeMLP: none) * (ReinitAndTrain: yes) 
}

plan attention_values {
  reach attention_values via (Dataset: ted) * (Lang: deen)
}

# Nuts and bolts:
global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}

