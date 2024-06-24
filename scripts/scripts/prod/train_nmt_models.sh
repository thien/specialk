cd ../..

# This is to be run on the azure machines.
# Note that this will utilise the whole sequence
# at token length 500, with BPE encodings.

# Preprocessing the dataset.
FILEPATH="../datasets/machine_translation/"
TRAIN_EN=$FILEPATH"corpus_enfr_final.train.en"
TRAIN_FR=$FILEPATH"corpus_enfr_final.train.fr"
VALID_EN=$FILEPATH"corpus_enfr_final.val.en"
VALID_FR=$FILEPATH"corpus_enfr_final.val.fr"
TEST_EN=$FILEPATH"corpus_enfr_final.test.en"
TEST_FR=$FILEPATH"corpus_enfr_final.test.fr"

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
 #       python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_FR -valid_src $VALID_EN -valid_tgt $VALID_FR -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_FR_ENFR  -vocab_size $VOCAB_SIZE

        python3 preprocess.py -train_src $TRAIN_FR -train_tgt $TRAIN_EN -valid_src $VALID_FR -valid_tgt $VALID_EN -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_FR_FREN  -vocab_size $VOCAB_SIZE
    fi
else
    echo "ERROR: Can't find $TRAIN_EN"
fi


# Train nmt models.

MODEL="transformer"
EP=3
MODELDIM=512
BATCHSIZE=45

ENFR_DIRNAME="fren_bpe_gold_master_mk2"
FREN_DIRNAME="fren_bpe_gold_master_mk2"

#python3 train.py \
#    -log $true \
#    -batch_size $BATCHSIZE \
#    -model $MODEL \
#    -epoch $EP \
#    -d_word_vec $MODELDIM \
#    -d_model $MODELDIM \
#    -data $SAVENAME_FR_ENFR$PTF \
#    -save_model \
#    -save_mode all \
#    -directory_name $ENFR_DIRNAME \
#    -cuda \
#    -multi_gpu

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
    -cuda
#    -multi_gpu
