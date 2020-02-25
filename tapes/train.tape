task train : scripts
    < train_src=$TRAIN_SRC@bpe_data
    < train_trg=$TRAIN_TRG@bpe_data
    < dev_src=$DEV_SRC@bpe_data
    < dev_trg=$DEV_TRG@bpe_data
    > model
    :: model_size=@
    :: ((pyenv=))@ use_cpu=@
    :: .submitter=@ .resource_flags=$resource_flags_train .action_flags=@ {

    $scripts/train_transformer.sh -o $model \
    --source $train_src \
    --target $train_trg \
    --validation-source $dev_src \
    --validation-target $dev_trg \
    --transformer-dropout-attention=0.1 \
    --transformer-dropout-act=0.1 \
    --transformer-dropout-prepost=0.1 \
    --label-smoothing=0.1 \
    $($scripts/size_params.sh $model_size)
}

task decode_dev calls decode 
    < in=$DEV_SRC@bpe_data
    < model=$model@train
    > out out_log out_scores
    :: use_cpu=@ pyenv=@
    :: .submitter=@ :: .action_flags=@ :: .resource_flags=$resource_flags_decode

task bleu_dev calls multi_bleu
     < in=$out@decode_dev
     < ref=$DEV_TRG@bpe_data
      > bleu