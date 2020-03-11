task FF_data : scripts
  > data_dir {
  python $scripts/mnist_data.py $data_dir
}

task train_FF : scripts
  < data_dir=$data_dir@FF_data
  > model
  :: lr=(FF_lr: 0.1 0.01 0.001)
  :: pyenv=@ use_cpu=@
  :: .submitter=@ .resource_flags=$resource_flags_train_FF .action_flags=@ {
  # Hyper-parameters can be adjusted, but we get decent validation acc so I'm not worried
  python $scripts/lenet.py [300,100] $model $data_dir --lr=$lr
}

task prune_FF : scripts
  < in_model=$model@train_FF
  > out_model
  :: sparsity=(SparsityFF: 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95)
  :: prune_type=(PruneTypeFF: Coreset Uniform Top-K) {
  python $scripts/prune_lenet.py $in_model $out_model $sparsity $prune_type
}

task eval_FF : scripts
  < data_dir=$data_dir@FF_data
  < in_model=$model@prune_FF
  :: pyenv=@ use_cpu=@
  :: .submitter=@ .resource_flags=$resource_flags .action_flags=@ {
  {
  python $scripts/prune_lenet.py $in_model $data_dir
}

task finetune_FF {}

task eval_finetune_FF {}