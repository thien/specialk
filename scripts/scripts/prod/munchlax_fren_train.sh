cd ../..

# This is to be run on the azure machines.
# Note that this will utilise the whole sequence
# at token length 500, with BPE encodings.

# Preprocessing the dataset.
FILEPATH="../datasets/machine_translation/"
TRAIN_EN=$FILEPATH"corpus_enfr_final_gm.train.en"
TRAIN_FR=$FILEPATH"corpus_enfr_final_gm.train.fr"
VALID_EN=$FILEPATH"corpus_enfr_final_gm.val.en"
VALID_FR=$FILEPATH"corpus_enfr_final_gm.val.fr"
TEST_EN=$FILEPATH"corpus_enfr_final_gm.test.en"
TEST_FR=$FILEPATH"corpus_enfr_final_gm.test.fr"

FORMAT="bpe"
MAXLEN="100"
PTF=".pt"
B="_base"
m="models/"
EXT="_bpe"
VOCAB_SIZE="35000"
FR_ENFR="nmt_enfr_goldmaster_mk2"$EXT
FR_FREN="nmt_fren_goldmaster_mk2"$EXT
SAVENAME_FR_ENFR=$m$FR_ENFR
SAVENAME_FR_FREN=$m$FR_FREN

if [ -e $TRAIN_EN ]
then
    if [ -e $SAVENAME_FR_FREN$PTF ]
    then
        echo "xl corpus already created."
    else
        # python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_FR -valid_src $VALID_EN -valid_tgt $VALID_FR -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_FR_ENFR  -vocab_size $VOCAB_SIZE

        python3 preprocess.py -train_src $TRAIN_FR -train_tgt $TRAIN_EN -valid_src $VALID_FR -valid_tgt $VALID_EN -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_FR_FREN  -vocab_size $VOCAB_SIZE
    fi
fi


# Train nmt models.

MODEL="transformer"
EP=3
MODELDIM=512
BATCHSIZE=45

# ENFR_DIRNAME="enfr_bpe_gold_master"

# python3 train.py \
#     -log $true \
#     -batch_size $BATCHSIZE \
#     -model $MODEL \
#     -epoch $EP \
#     -d_word_vec $MODELDIM \
#     -d_model $MODELDIM \
#     -data $SAVENAME_FR_ENFR$PTF \
#     -save_model \
#     -save_mode all \
#     -directory_name $ENFR_DIRNAME \
#     -cuda \
#     -multi_gpu

# python3 core/telegram.py -m "[Snorlax] Finished training gold master en-fr models."

FREN_DIRNAME="fren_bpe_gold_master_mk2"

python3 train.py \
    -log $true \
    -batch_size $BATCHSIZE \
    -model $MODEL \
    -epoch $EP \
    -d_word_vec $MODELDIM \
    -d_model $MODELDIM \
    -data $SAVENAME_FR_FREN$PTF \
    -save_model \
    -save_mode all \
    -directory_name $FREN_DIRNAME \
    -cuda \
    -checkpoint_encoder "models/"$FREN_DIRNAME"/encoder_epoch_1.chkpt" \
    -checkpoint_decoder "models/"$FREN_DIRNAME"/decoder_epoch_1.chkpt"

python3 core/telegram.py -m "[Snorlax] Finished training gold master fr-en models."

# # Back-Translate nmt models.

# ENFR_BASEDIR="models/"$ENFR_DIRNAME"/"
# FREN_BASEDIR="models/"$ENFR_DIRNAME"/"
# ENCODER='encoder.chkpt'
# DECODER='decoder.chkpt'
# OUTPUT="outputs.txt"
# EVALTXT="eval.txt"

# # en -> fr
# python3 translate.py \
#     -model $MODEL \
#     -checkpoint_encoder $ENFR_BASEDIR$ENCODER \
#     -checkpoint_decoder $ENFR_BASEDIR$DECODER \
#     -vocab $SAVENAME_FR_ENFR$PTF \
#     -src $TEST_EN \
#     -output $ENFR_BASEDIR$OUTPUT \
#     -cuda

# python3 core/telegram.py -m "[Snorlax] Finished translating en-fr models."

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
