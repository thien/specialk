cd ../..


VOCAB="models/nmt_enfr_bpe_filtered"
FORMAT="bpe"
MAXLEN="150"

FILEPATH="../datasets/machine_translation/corpus_enfr_filtered"

# setup dataset preprocessing.
PT=".pt"




TRAIN_EN=$FILEPATH".train.en"
TRAIN_DE=$FILEPATH".train.fr"
VALID_EN=$FILEPATH".val.en"
VALID_DE=$FILEPATH".val.fr"

# make the corpus.


if [ -e $VOCAB".pt" ]
then
    echo "corpus already made."
else
    python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_DE -valid_src $VALID_EN -valid_tgt $VALID_DE -format $FORMAT -max_len $MAXLEN -save_name $VOCAB -max_train_seq 50000
fi

# TRAIN
MODEL="transformer"
DIRNAME="enfr_test_bpe_filtered_baby"
EP=10
MODELDIM=512
BATCHSIZE=32
python3 train.py -data $VOCAB$PT -log $true -save_model -model $MODEL -epoch $EP -d_word_vec $MODELDIM -d_model $MODELDIM -save_mode "best" -directory_name $DIRNAME -batch_size $BATCHSIZE -cuda 

# python3 train.py -data $VOCAB$PT -log $true -model $MODEL -epoch $EP -d_word_vec $MODELDIM -d_model $MODELDIM -cuda -batch_size $BATCHSIZE

# # TRANSLATE
TESTDATA="../datasets/machine_translation/corpus_enfr_filtered.en.atok"

BASEDIR="models/"$DIRNAME"/"
ENCODER='encoder.chkpt'
DECODER='decoder.chkpt'
OUTPUT="outputs.txt"
EVALTXT="eval.txt"

python3 translate.py -model $MODEL -checkpoint_encoder $BASEDIR$ENCODER -checkpoint_decoder $BASEDIR$DECODER -vocab $VOCAB$PT -src $TESTDATA -output $BASEDIR$OUTPUT -batch_size 10

# # evaluate performance
# nlg-eval --hypothesis=$BASEDIR$OUTPUT --references=$TESTDATA > $BASEDIR$EVALTXT
