for l in en fr; do for f in ../data/ccep/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
for l in en fr; do for f in ../data/ccep/*.$l; do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok; done; done
python preprocess.py -train_src ../data/ccep/train_ccep.en.atok -train_tgt ../data/ccep/train_ccep.fr.atok -valid_src ../data/ccep/val_ccep.en.atok -valid_tgt ../data/ccep/val_ccep.fr.atok -save_data ../data/ccep/ccep.atok.low -lower