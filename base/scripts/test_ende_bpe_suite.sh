cd ..


VOCAB="models/nmt_ende_bpe_new"
FORMAT="bpe"
MAXLEN="150"


# setup dataset preprocessing.
PT=".pt"
if [ -e $VOCAB$PT ]
then 
    echo "Dataset already preprocessed."
    rm $VOCAB$PT 
fi
# else
# make the corpus.
if [ -e ../datasets/multi30k/train.en.atok ]
then
    FILEPATH="../datasets/multi30k/"
    TRAIN_EN=$FILEPATH"train.en.atok"
    TRAIN_DE=$FILEPATH"train.de.atok"
    VALID_EN=$FILEPATH"val.en.atok"
    VALID_DE=$FILEPATH"val.de.atok"

    python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_DE -valid_src $VALID_EN -valid_tgt $VALID_DE -format $FORMAT -max_len $MAXLEN -save_name $VOCAB
else
    echo "You need to create the corpus."
fi
fi

# TRAIN
MODEL="transformer"
DIRNAME="ende_test_bpe"
EP=15
MODELDIM=512
BATCHSIZE=64
python3 train.py -data $VOCAB$PT -log $true -save_model -model $MODEL -epoch $EP -d_word_vec $MODELDIM -d_model $MODELDIM -save_mode "best" -directory_name $DIRNAME -batch_size $BATCHSIZE -cuda 

# python3 train.py -data $VOCAB$PT -log $true -model $MODEL -epoch $EP -d_word_vec $MODELDIM -d_model $MODELDIM -cuda -batch_size $BATCHSIZE

# # TRANSLATE
TESTDATA="../datasets/multi30k/test.en.atok"

BASEDIR="models/"$DIRNAME"/"
ENCODER='encoder_.chkpt'
DECODER='decoder_.chkpt'
OUTPUT="outputs.txt"
EVALTXT="eval.txt"

python3 translate.py -model $MODEL -checkpoint_encoder $BASEDIR$ENCODER -checkpoint_decoder $BASEDIR$DECODER -vocab $VOCAB$PT -src $TESTDATA -output $BASEDIR$OUTPUT -cuda

# # evaluate performance
nlg-eval --hypothesis=$BASEDIR$OUTPUT --references=$TESTDATA > $BASEDIR$EVALTXT
