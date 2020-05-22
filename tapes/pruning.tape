task prune_attn : scripts
    :: prune_type=(PruneType: uniform topk L1 coreset randmatmul)
    :: trial=(Trial: 1..5)
    :: sparsity=(Sparsity: 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
    < model=$model@train
    > new_attn_size
    > out {
    python $scripts/prune_transformer_attn.py $model/checkpoint_best.pt $out $sparsity $prune_type > $new_attn_size
}

task maybe_prune_mlp : scripts
    :: prune_type=(PruneTypeMLP: none uniform dataind_coreset datadep_coreset)
    :: sparsity=$sparsity@prune_attn
    < model=$out@prune_attn
    > new_mlp_size
    > out {
    if [ $prune_type == none ]; then
      echo "Not pruning MLPs"
      ln -s $model $out
      echo 2048 > new_mlp_size
    elif [ $prune_type == uniform ]; then
      python $scripts/prune_transformer_MLP.py $model $out $sparsity > $new_mlp_size
    fi
}

task bleu_dev_post_prune
    < in=$prepped_data@preprocess_data
    < model=$out@maybe_prune_mlp
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

summary post_prune {
  of bleu_dev_post_prune > post_prune {
    cat $out | tail -n 1 | cut -d' ' -f 3 | sed 's/.$//' > $post_prune
  }
}