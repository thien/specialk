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


# Train nmt models.

EP=15
MODELDIM=512
BATCHSIZE=64

FREN_DIRNAME="fren_bpe"

python3 train.py \
    -log $true \
    -batch_size $BSIZE \
    -model $MODEL \
    -epoch $EP \
    -d_word_vec $MODELDIM \
    -d_model $MODELDIM \
    -data $SAVENAME_FR_FREN$PTF \
    -save_model \
    -save_mode best \
    -directory_name FREN_DIRNAME \
    -cuda \
    -cuda_device 1

python3 core/telegram.py -m "[Snorlax] Finished training fr-en models."

# # created fr->en
# python3 translate.py \
#     -model $MODEL \
#     -checkpoint_encoder $FREN_BASEDIR$ENCODER \
#     -checkpoint_decoder $FREN_BASEDIR$DECODER \
#     -vocab $SAVENAME_FR_FREN$PTF \
#     -src $ENFR_BASEDIR$OUTPUT \
#     -output $FREN_BASEDIR$OUTPUT \
#     -cuda

# python3 core/telegram.py -m "[Snorlax] Finished translating fr-en (backtranslation) models."

# # baseline fr->en
# python3 translate.py \
#     -model $MODEL \
#     -checkpoint_encoder $FREN_BASEDIR$ENCODER \
#     -checkpoint_decoder $FREN_BASEDIR$DECODER \
#     -vocab $SAVENAME_FR_FREN$PTF \
#     -src $TEST_FR \
#     -output $FREN_BASEDIR$OUTPUT \
#     -cuda

# python3 core/telegram.py -m "[Snorlax] Finished translating fr-en (source) models."
