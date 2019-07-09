cd ..

if [ -e ../datasets/machine_translation/corpus_enfr.test.en ]
then
    FILEPATH="../datasets/machine_translation/"
    TRAIN_EN=$FILEPATH"corpus_enfr.train.en"
    TRAIN_FR=$FILEPATH"corpus_enfr.train.fr"
    VALID_EN=$FILEPATH"corpus_enfr.val.en"
    VALID_FR=$FILEPATH"corpus_enfr.val.fr"
    FORMAT="word"
    MAXLEN="70"

    # # setup small corpus
    SM="50000"
    SAVENAME_SM="models/nmt_enfr_50k"
    python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_FR -valid_src $VALID_EN -valid_tgt $VALID_FR -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_SM -max_train_seq $SM

    # setup medium corpus
    MD="500000"
    SAVENAME_MD="models/nmt_enfr_500k"
    python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_FR -valid_src $VALID_EN -valid_tgt $VALID_FR -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_MD -max_train_seq $MD

    # setup large corpus
    LG="1500000"
    SAVENAME_LG="models/nmt_enfr_lg"
    python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_FR -valid_src $VALID_EN -valid_tgt $VALID_FR -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_LG -max_train_seq $LG

    # setup full
    SAVENAME_FULL="models/nmt_enfr_full"
    python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_FR -valid_src $VALID_EN -valid_tgt $VALID_FR -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_FULL
else
    echo "You need to create the corpus."
fi