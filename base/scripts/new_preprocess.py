cd ..

clear

VOCAB="models/nmt_ende_bpe_a"
FORMAT="word"
MAXLEN="100"

FILEPATH="../datasets/multi30k/"
TRAIN_EN=$FILEPATH"train.en.atok"
TRAIN_DE=$FILEPATH"train.de.atok"
VALID_EN=$FILEPATH"val.en.atok"
VALID_DE=$FILEPATH"val.de.atok"
PT=".pt"

# setup dataset preprocessing.
if [ -e $VOCAB$PT ]
then 
    echo "Dataset already preprocessed."
    rm $VOCAB$PT 
fi
# else
# make the corpus.
if [ -e $TRAIN_EN ]
then
    python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_DE -valid_src $VALID_EN -valid_tgt $VALID_DE -format $FORMAT -max_len $MAXLEN -save_name $VOCAB
else
    echo "You need to create the corpus."
fi

VOCAB="models/nmt_ende_bpe_b"

# setup dataset preprocessing.
if [ -e $VOCAB$PT ]
then 
    echo "Dataset already preprocessed."
    rm $VOCAB$PT 
fi
# else
# make the corpus.
if [ -e $TRAIN_EN ]
then
    python3 preprocess_new.py -train_src $TRAIN_EN -train_tgt $TRAIN_DE -valid_src $VALID_EN -valid_tgt $VALID_DE -format $FORMAT -max_len $MAXLEN -save_name $VOCAB
else
    echo "You need to create the corpus."
fi