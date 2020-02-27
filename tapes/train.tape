task train : scripts
    < prepped_data=$prepped_data@preprocess_data
    > model
    :: model_size=@
    :: pyenv=@ use_cpu=@ logdir=@
    :: .submitter=@ .resource_flags=$resource_flags_train .action_flags=@ {

    fairseq-train $prepped_data \
    --save-dir $model \
    --arch transformer \
    --encoder-learned-pos \
    --decoder-learned-pos \
    --attention-dropout 0.1 \
    --activation-dropout 0.1 \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --optimizer adam \
    --lr-scheduler reduce_lr_on_plateau \
    --lr 0.0002 \
    --lr-shrink 0.9 \
    --max-epoch 100 \
    --max-update 300000 \
    --patience 6 \
    --keep-last-epochs 6 \
    --criterion label_smoothed_cross_entropy \
    --label-smoothing 0.1 \
    --clip-norm 1 \
    --max-tokens 4096 \
    --tensorboard-logdir $logdir
    # TODO: adjust patience and validation interval depending on the dataset


    # Should use warmup-initial-lr or lr? Was previously initial-learning-rate
    # Pre/post processing blocks / dropout??

    #   --learning-rate-reduce-num-not-improved=8 \
    # This is num of checkpoints not improved before lr decreases
    # Corresponds to patience in pytorch.optim.lr_scheduler.ReduceLROnPlateau, (epochs not checkpoints)
    # fairseq sets this to 0 by default (and doesn't let you change it.) https://git.io/Jvz4N

    #   --learning-rate-decay-optimizer-states-reset=best \
    # Same story here. This resets Adam when the learning rate is decreased. No equiv in fairseq

    #   --learning-rate-decay-param-reset \
    # And here. Resets model params to last best when lr decreased. No equiv.

    # --weight-init=xavier \
    # --weight-init-scale=3.0 \
    # --weight-init-xavier-factor-type=avg \
    # Xavier in fairseq is fixed depending on where its used. No config available.

    # --optimized-metric=perplexity \

    # Residual connections are assumed.

    # Note: if you find yourself adding back in a branch for model size, consider resurrecting scripts/size_params.sh
}

task bleu_dev
    < in=$prepped_data@preprocess_data
    < model=$model@train
    > out
    :: use_cpu=@ pyenv=@
    :: .submitter=@ :: .action_flags=@ :: .resource_flags=$resource_flags_decode {

    fairseq-generate $in --path $model/checkpoint_best.pt \
    --batch-size 8 \
    --gen-subset valid \
    --beam 12 > out.all 
    # --max-input-len 300 \

    grep ^H out.all | cut -f3- > gen.out.sys
    grep ^T out.all | cut -f2- > gen.out.ref
    fairseq-score --sys gen.out.sys --ref gen.out.ref > out
}