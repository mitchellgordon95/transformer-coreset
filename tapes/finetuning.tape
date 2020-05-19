task finetune : scripts
    < prepped_data=$prepped_data@preprocess_data
    < init_model=$out@maybe_prune_mlp
    < new_attn_size=$new_attn_size@prune_attn
    < new_mlp_size=$new_mlp_size@maybe_prune_mlp
    > model
    :: share_qk=$share_qk@train
    :: reinit_and_train=(ReinitAndTrain: no yes)
    :: model_size=@
    :: pyenv=@ use_cpu=@ logdir=@
    :: .submitter=@ .resource_flags=$resource_flags_train .action_flags=@ {

  if [ $reinit_and_train == no ]; then
     params="--restore-file $init_model 
        --reset-optimizer 
        --reset-dataloader 
        --reset-meters 
        --reset-lr-scheduler 
        --max-epoch 1 
        --max-update 3000 "
  else
     params="--max-epoch 100 --max-update 300000"
  fi

  if [ $share_qk == "yes" ]; then
    share_qk="--shared-qk"
  else
    share_qk=""
  fi

  $scripts/train_ted.sh $prepped_data $share_qk \
  --save-dir $model \
  --encoder-attn-proj-dim $(head -n 1 $new_attn_size) \
  --decoder-attn-proj-dim $(tail -n 1 $new_attn_size) \
  --encoder-ffn-embed-dim $(head -n 1 $new_mlp_size) \
  --decoder-ffn-embed-dim $(tail -n 1 $new_mlp_size) \
  --tensorboard-logdir logdir \
  $params
}

task bleu_dev_post_finetune
    < in=$prepped_data@preprocess_data
    < model=$model@finetune
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

summary post_finetune {
  of bleu_dev_post_finetune > post_finetune {
    cat $out | tail -n 1 | cut -d' ' -f 3 | sed 's/.$//' > $post_finetune
  }
}