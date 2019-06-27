if [ -e models/nmt_ende.pt ]
then 
    clear
    # train the model.
    # DATA="models/nmt_ende.pt"
    # echo "ready"
    # MODEL="transformer"
    # EP=15
    # MODELDIM=512
    # python3 train.py -data $DATA -log $true -save_model -model $MODEL -epoch $EP -d_word_vec $MODELDIM -d_model $MODELDIM -cuda

    DATA="models/nmt_ende.pt"
    echo "ready"
    MODEL="recurrent"
    EP=15
    MODELDIM=512
    python3 train.py -data $DATA -log $true -save_model -model $MODEL -epoch $EP -d_word_vec $MODELDIM -d_model $MODELDIM -cuda

else
    # make the corpus.
    if [ -e ../datasets/multi30k/train.en.atok ]
    then
        FILEPATH="../datasets/multi30k/"
        TRAIN_EN=$FILEPATH"train.en.atok"
        TRAIN_DE=$FILEPATH"train.de.atok"
        VALID_EN=$FILEPATH"val.en.atok"
        VALID_DE=$FILEPATH"val.de.atok"
        FORMAT="word"
        MAXLEN="70"


        # setup full
        SAVENAME_FULL="models/nmt_ende"
        python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_DE -valid_src $VALID_EN -valid_tgt $VALID_DE -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_FULL
    else
        echo "You need to create the corpus."
    fi
fi