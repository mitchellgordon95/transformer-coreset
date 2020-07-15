task train : scripts
    < prepped_data=$prepped_data@preprocess_data
    > model
    :: share_qk=(ShareQK: no yes)
    :: model_size=@
    :: pyenv=@ use_cpu=@
    :: .submitter=@ .resource_flags=$resource_flags_train .action_flags=@ {

  if [ $share_qk == "yes" ]; then
    share_qk="--shared-qk"
  else
    share_qk=""
  fi

  $scripts/train_ted.sh $prepped_data $share_qk \
  --save-dir $model \
  --max-epoch 100 \
  --max-update 300000 \
  --tensorboard-logdir logdir

}


task bleu_dev
    < in=$prepped_data@preprocess_data
    < model=$model@train
    > out
    :: use_cpu=@ pyenv=@
    :: .submitter=@ :: .action_flags=@ :: .resource_flags=$resource_flags_decode {

    fairseq-generate $in --path $model/checkpoint_best.pt \
    --batch-size 10 \
    --gen-subset valid \
    --beam 12 > out.all 
    # --max-input-len 300 \

    grep ^H out.all | cut -f3- > gen.out.sys
    grep ^T out.all | cut -f2- > gen.out.ref
    fairseq-score --sys gen.out.sys --ref gen.out.ref > out
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

 