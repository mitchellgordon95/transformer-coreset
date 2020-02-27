task maybe_prune_chkpt : scripts
    :: method=(PruningMethod: coreset)
    < model=$model@train
    > out {
    # TODO: implement coreset pruning
}