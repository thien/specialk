# This is to be run on the azure machines.
# Note that this will utilise the whole sequence
# at token length 500, with BPE encodings.

# Preprocessing the dataset.
FILEPATH="../datasets/machine_translation/"
TRAIN_EN=$FILEPATH"corpus_enfr.train.en"
TRAIN_FR=$FILEPATH"corpus_enfr.train.fr"
VALID_EN=$FILEPATH"corpus_enfr.val.en"
VALID_FR=$FILEPATH"corpus_enfr.val.fr"
TEST_EN=$FILEPATH"corpus_enfr.test.en"
TEST_FR=$FILEPATH"corpus_enfr.test.fr"

FORMAT="bpe"
MAXLEN="500"
PTF=".pt"
B="_base"
m="models/"
EXT="_lower_4k_v_bpe"
VOCAB_SIZE="4096"
FR_ENFR="nmt_enfr_full"$EXT
FR_FREN="nmt_fren_full"$EXT
SAVENAME_FR_ENFR=$m$FR_ENFR
SAVENAME_FR_FREN=$m$FR_FREN

if [ -e ../datasets/machine_translation/corpus_enfr.test.en ]
then
    if [ -e $SAVENAME_FR_FREN$PTF ]
    then
        echo "xl corpus already created."
    else
        python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_FR -valid_src $VALID_EN -valid_tgt $VALID_FR -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_FR_ENFR  -vocab_size $VOCAB_SIZE

        python3 preprocess.py -train_src $TRAIN_FR -train_tgt $TRAIN_EN -valid_src $VALID_FR -valid_tgt $VALID_EN -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_FR_FREN  -vocab_size $VOCAB_SIZE
    fi
fi


# Train nmt models.

EP=15
MODELDIM=512
BATCHSIZE=64

ENFR_DIRNAME="enfr_bpe"

python3 train.py \
    -log $true \
    -batch_size $BSIZE \
    -model $MODEL \
    -epoch $EP \
    -d_word_vec $MODELDIM \
    -d_model $MODELDIM \
    -cuda \
    -data $SAVENAME_FR_ENFR$PTF \
    -save_model \
    -save_mode best \
    -directory_name $ENFR_DIRNAME


python3 core/telegram.py -m "[Snorlax] Finished training en-fr models."

FREN_DIRNAME="fren_bpe"

python3 train.py \
    -log $true \
    -batch_size $BSIZE \
    -model $MODEL \
    -epoch $EP \
    -d_word_vec $MODELDIM \
    -d_model $MODELDIM \
    -cuda \
    -data $SAVENAME_FR_FREN$PTF \
    -save_model \
    -save_mode best \
    -directory_name FREN_DIRNAME

python3 core/telegram.py -m "[Snorlax] Finished training fr-en models."

# Back-Translate nmt models.

ENFR_BASEDIR="models/"$ENFR_DIRNAME"/"
FREN_BASEDIR="models/"$ENFR_DIRNAME"/"
ENCODER='encoder_.chkpt'
DECODER='decoder_.chkpt'
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
    -cuda

python3 core/telegram.py -m "[Snorlax] Finished translating en-fr models."

# created fr->en
python3 translate.py \
    -model $MODEL \
    -checkpoint_encoder $FREN_BASEDIR$ENCODER \
    -checkpoint_decoder $FREN_BASEDIR$DECODER \
    -vocab $SAVENAME_FR_FREN$PTF \
    -src $ENFR_BASEDIR$OUTPUT \
    -output $FREN_BASEDIR$OUTPUT \
    -cuda

python3 core/telegram.py -m "[Snorlax] Finished translating fr-en (backtranslation) models."

# baseline fr->en
python3 translate.py \
    -model $MODEL \
    -checkpoint_encoder $FREN_BASEDIR$ENCODER \
    -checkpoint_decoder $FREN_BASEDIR$DECODER \
    -vocab $SAVENAME_FR_FREN$PTF \
    -src $TEST_FR \
    -output $FREN_BASEDIR$OUTPUT \
    -cuda

python3 core/telegram.py -m "[Snorlax] Finished translating fr-en (source) models."
