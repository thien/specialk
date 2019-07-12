cd ..

if [ -e models/nmt_ende.pt ]
then 
    echo "Dataset already preprocessed."
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

# TRAIN
DATA="models/nmt_ende.pt"
echo "ready"
MODEL="transformer"
EP=15
MODELDIM=512
python3 train.py -data $DATA -log $true -save_model -model $MODEL -epoch $EP -d_word_vec $MODELDIM -d_model $MODELDIM -cuda

# TRANSLATE
# TESTDATA="../datasets/multi30k/test.en.atok"
# VOCAB="models/nmt_ende.pt"
# BASEDIR="models/transformer-19-06-26-18-12-36/"
# ENCODER='encoder_epoch_14_accu_51.533.chkpt'
# DECODER='decoder_epoch_14_accu_51.533.chkpt'
# OUTPUT="outputs.txt"
# EVALTXT="eval.txt"

# python3 translate.py -model $MODEL -checkpoint_encoder $BASEDIR$ENCODER -checkpoint_decoder $BASEDIR$DECODER -vocab $VOCAB -src $TESTDATA -output $BASEDIR$OUTPUT -cuda

# nlg-eval --hypothesis=$BASEDIR$OUTPUT --references=$TESTDATA > $BASEDIR$EVALTXT

# # -----------------------
# # RECURRENT
# # -----------------------

# # DATA="models/nmt_ende.pt"
# # echo "ready"
# MODEL="recurrent"
# # # EP=5
# # # BATCHSIZE=64
# # # MODELDIM=512
# # # EMB=512
# # # python3 train.py -data $DATA -log $true -model $MODEL -epoch $EP -d_word_vec $EMB -d_model $MODELDIM -cuda -layers 2 -batch_size $BATCHSIZE -save_model

# ENCODER='models/recurrent-19-07-01-19-57-44/encoder_epoch_5_accu_99.921.chkpt'
# DECODER='models/recurrent-19-07-01-19-57-44/decoder_epoch_5_accu_99.921.chkpt'
# OUTPUT="models/recurrent-19-07-01-19-57-44/outputs.txt"
# # python translate.py -model trained.chkpt -vocab data/multi30k.atok.low.pt -src data/multi30k/test.en.atok -no_cuda
# python3 translate.py -model $MODEL -checkpoint_encoder $ENCODER -checkpoint_decoder $DECODER -vocab $VOCAB -src $TESTDATA -output $OUTPUT -cuda
 
 