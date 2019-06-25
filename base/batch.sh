if [ -e ../datasets/machine_translation/corpus_enfr.test.en ]
then
    FILEPATH="../datasets/machine_translation/"
    TRAIN_EN=$FILEPATH"corpus_enfr.train.en"
    TRAIN_FR=$FILEPATH"corpus_enfr.train.fr"
    VALID_EN=$FILEPATH"corpus_enfr.val.en"
    VALID_FR=$FILEPATH"corpus_enfr.val.fr"
    FORMAT="word"
    MAXLEN="70"

    SAVEDATA="nmt_enfr"
    python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_FR -valid_src $VALID_EN -valid_tgt $VALID_FR -format $FORMAT -save_data $SAVEDATA
else
    echo "You need to create the corpus."
fi


