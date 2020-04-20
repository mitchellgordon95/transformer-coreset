task prune : scripts
    :: prune_type=(PruneType: uniform dataind_coreset datadep_coreset)
    :: trial=(Trial: 1..5)
    :: sparsity=(Sparsity: 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 0.95)
    < model=$model@train
    > out {
    python $scripts/prune_transformer_MLP.py $model/checkpoint_best.pt tmp $sparsity
    python $scripts/prune_transformer_attn.py tmp $out $sparsity $prune_type
}

task bleu_dev_post_prune
    < in=$prepped_data@preprocess_data
    < model=$out@prune
    > out
    :: use_cpu=@ pyenv=@
    :: .submitter=@ :: .action_flags=@ :: .resource_flags=$resource_flags_decode {

    fairseq-generate $in --path $model \
    --batch-size 10 \
    --gen-subset valid \
    --beam 12 > out.all 
    # --max-input-len 300 \

    grep ^H out.all | cut -f3- > gen.out.sys
    grep ^T out.all | cut -f2- > gen.out.ref
    fairseq-score --sys gen.out.sys --ref gen.out.ref > out
}