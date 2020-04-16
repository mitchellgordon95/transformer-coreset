task prune_mlp : scripts
    :: prune_type=(PruneTypeMLP: uniform dataind_coreset datadep_coreset)
    :: trial=(Trial: 1..5)
    :: sparsity=(Sparsity: 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95)
    < model=$model@train
    > out {
    python $scripts/prune_transformer_mlp/uniform.py $model/checkpoint_best.pt $out $sparsity
}

task prune_attn : scripts
    :: prune_type=(PruneTypeAttn: uniform dataind_coreset datadep_coreset)
    :: sparsity=$sparsity@prune_mlp
    < model=$out@prune_mlp
    > out {
    # TODO: implement coreset pruning
}

task bleu_dev_post_prune
    < in=$prepped_data@preprocess_data
    < model=$out@prune_attn
    > out
    :: use_cpu=@ pyenv=@
    :: .submitter=@ :: .action_flags=@ :: .resource_flags=$resource_flags_decode {

    fairseq-generate $in --path $model/checkpoint_best.pt \
    --batch-size 10 \
    --gen-subset valid \
    --model-overrides "{\"attention_type\":\"$attn_type\"}"\
    --beam 12 > out.all 
    # --max-input-len 300 \

    grep ^H out.all | cut -f3- > gen.out.sys
    grep ^T out.all | cut -f2- > gen.out.ref
    fairseq-score --sys gen.out.sys --ref gen.out.ref > out
}