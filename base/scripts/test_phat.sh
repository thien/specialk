cd ..

# Preprocessing the dataset.
FILEPATH="../datasets/machine_translation/"
TRAIN_EN=$FILEPATH"corpus_enfr.train.en"
TRAIN_FR=$FILEPATH"corpus_enfr.train.fr"
VALID_EN=$FILEPATH"corpus_enfr.val.en"
VALID_FR=$FILEPATH"corpus_enfr.val.fr"
TEST_EN=$FILEPATH"corpus_enfr.test.en"
TEST_FR=$FILEPATH"corpus_enfr.test.fr"

FORMAT="bpe"
MAXLEN="200"
PTF=".pt"
B="_base"
m="models/"
EXT="_lower_4k_v_bpe"
VOCAB_SIZE="4096"
FR_ENFR="nmt_enfr_full"$EXT
FR_FREN="nmt_fren_full"$EXT
SAVENAME_FR_ENFR=$m$FR_ENFR
SAVENAME_FR_FREN=$m$FR_FREN



EP=5
MODELDIM=512
BATCHSIZE=64
MODEL="transformer"

ENFR_DIRNAME="phat_enfr_bpe"

ENFR_BASEDIR="models/"$ENFR_DIRNAME"/"
FREN_BASEDIR="models/"$ENFR_DIRNAME"/"
ENCODER='encoder.chkpt'
DECODER='decoder.chkpt'
OUTPUT="outputs.txt"
EVALTXT="eval.txt"

# en -> fr
python3 translate.py \
    -model $MODEL \
    -checkpoint_encoder $ENFR_BASEDIR$ENCODER \
    -checkpoint_decoder $ENFR_BASEDIR$DECODER \
    -vocab $SAVENAME_FR_ENFR$PTF \
    -src $TEST_EN \
    -output $ENFR_BASEDIR$OUTPUT \
    -batch_size $BATCHSIZE \
    -cuda


