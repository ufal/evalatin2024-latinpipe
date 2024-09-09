# The `latinpipe-evalatin24-240520` Model

The `latinpipe-evalatin24-240520` is a `PhilBerta`-based model for tagging,
lemmatization, and dependency parsing of Latin, based on the winning entry
to the EvaLatin 2024 <https://circse.github.io/LT4HALA/2024/EvaLatin> shared
task. It is released at https://hdl.handle.net/11234/1-5671 under the CC
BY-NC-SA 4.0 license.

The model is also available in the [UDPipe LINDAT/CLARIN service](http://lindat.mff.cuni.cz/services/udpipe/)
 and can be used either in a web form or through a REST service.

The model was trained using the following command:
```sh
la_ud213_all="la_ittb la_llct la_perseus la_proiel la_udante"
la_other="la_archimedes la_sabellicus"
transformer="bowphs/PhilBerta"  # or bowphs/LaBerta

latinpipe_evalatin24.py $(for split in dev test train; do echo --$split; for tb in $la_ud213_all; do [ $tb-$split = la_proiel-train ] && tb=la_proielh; echo data/$tb/$tb-ud-$split.conllu; done; done) $(for tb in $la_other; do echo data/$tb/$tb-train.conllu; done) --transformers $transformer --epochs=30 --exp=latinpipe-evalatin24-240520 --subword_combination=last --epochs_frozen=10 --batch_size=64 --save_checkpoint
```

## EvaLatin 2024 LAS Results

The model achieves the following EvaLatin 2024 results (measured by the
official scorer). Note that the results are worse than in the paper, where the
model used the gold POS tags and was ensembled.

Treebank              | UAS   | LAS   |
:-------------------- |:-----:|:-----:|
EvaLatin 2024 Poetry  | 78.31 | 72.36 |
EvaLatin 2024 Prose   | 80.49 | 75.20 |
EvaLatin 2024 Average | 79.40 | 73.78 |

## UD 2.13 Results

The model achieves the following UPOS, UFeats, Lemmas, UAS, LAS metrics (as
measured by the official scorer) on the UD 2.13 test data. Note that the model
was trained on harmonized PROIEL, so we do not include the official PROIEL
treebank in the evaluation.

Treebank | UPOS  | UFeats | Lemmas | UAS   | LAS   |
:--------|:-----:|:------:|:------:|:-----:|:-----:|
ITTB     | 99.30 | 98.53  | 98.64  | 93.82 | 92.31 |
LLCT     | 99.80 | 97.60  | 96.39  | 96.37 | 95.35 |
Perseus  | 96.01 | 91.14  | 88.88  | 87.93 | 82.50 |
UDante   | 93.73 | 89.27  | 87.76  | 83.97 | 78.67 |

## Predicting with the `latinpipe-evalatin24-240520` Model

To predict with the `latinpipe-evalatin24-240520` model, you can use the following command:
```sh
latinpipe_evalatin24.py --load latinpipe-evalatin24-240520/model.weights.h5 --exp target_directory --test input1.conllu input2.conllu
```
- the outputs are generated in the target directory, with a `.predicted.conllu` suffix;
- if you want to also evaluate the predicted files, you can use `--dev` option instead of `--test`.

## How to Cite

```
@inproceedings{straka-etal-2024-ufal,
    title = "{{\'U}FAL} {L}atin{P}ipe at {E}va{L}atin 2024: Morphosyntactic Analysis of {L}atin",
    author = "Straka, Milan  and Strakov{\'a}, Jana  and Gamba, Federica",
    editor = "Sprugnoli, Rachele  and Passarotti, Marco",
    booktitle = "Proceedings of the Third Workshop on Language Technologies for Historical and Ancient Languages (LT4HALA) @ LREC-COLING-2024",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lt4hala-1.24",
    pages = "207--214"
}
```
