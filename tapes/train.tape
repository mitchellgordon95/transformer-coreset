task train : scripts
    < prepped_data=$prepped_data@preprocess_data
    > model
    :: model_size=@
    :: pyenv=@ use_cpu=@ logdir=@
    :: .submitter=@ .resource_flags=$resource_flags_train .action_flags=@ {

  $scripts/train_ted.sh $prepped_data \
  --save-dir $model \
  --max-epoch 100 \
  --max-update 300000 \
  --tensorboard-logdir $logdir

}

# task attention_values
#     < in=$prepped_data@preprocess_data
#     < model=$model@train
#     :: use_cpu=@ pyenv=@
#     :: .submitter=@ :: .action_flags=@ :: .resource_flags=$resource_flags_decode {
#     # Recall that we had some code in fairseq to make this work.
#     mkdir tmp
#     fairseq-generate $in --path $model/checkpoint_best.pt \
#     --batch-size 8 \
#     --gen-subset valid
#     # --max-input-len 300 \
# }

 