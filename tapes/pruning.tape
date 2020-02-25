task maybe_prune_chkpt : scripts
    :: (PruningMethod: coreset)
    < model=$model@train
    > out {
    # TODO: implement coreset pruning
}