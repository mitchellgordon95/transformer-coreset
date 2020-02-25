import "tapes/submitters.tape"
import "tapes/versioners.tape"
import "tapes/sockeye.tape"
import "tapes/bleu.tape"

## Tasks start here
import "tapes/data_prep.tape"
import "tapes/teacher.tape"
import "tapes/pruning.tape"
import "tapes/finetuning.tape"

plan all {
  reach teacher_bleu_dev via (Dataset: *) * (Lang: *) * (ModelSize: large)
}

# Nuts and bolts:
global {
  ducttape_experimental_packages=true
  ducttape_experimental_submitters=true
  ducttape_experimental_imports=true
  ducttape_experimental_multiproc=true
}

