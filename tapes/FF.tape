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
  :: beta=(BetaFF: 1 45)
  :: trial=(TrialFF: 1..5)
  :: sparsity=(SparsityFF: 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95)
  :: prune_type=(PruneTypeFF: Coreset Uniform Top-K) {
  python $scripts/prune_lenet.py $in_model $out_model $sparsity $prune_type $beta
}

task eval_FF : scripts
  < data_dir=$data_dir@FF_data
  < in_model=$out_model@prune_FF
  > score
  :: pyenv=@ use_cpu=@
  :: .submitter=@ .resource_flags=$resource_flags .action_flags=@ {
  python $scripts/score_lenet.py $in_model $data_dir > $score
}

task agg_eval_FF
  < scores=$score@eval_FF[SparsityFF:*]
  :: sparsities=$sparsity@prune_FF[SparsityFF:*]
  > sparsity_vs_acc
  {
  for score_f in $scores; do
    cat $score_f | tail -n 1 | cut -d' ' -f 4 >> scores_out
  done
  for sparsity in $sparsities; do
    echo $sparsity >> sparsities_out
  done
  paste sparsities_out scores_out > $sparsity_vs_acc
}

task finetune_FF {}

task eval_finetune_FF {}