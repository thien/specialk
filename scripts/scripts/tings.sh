cd ..

MODEL="transformer"

m="models/"

SM_ENFR="nmt_enfr_50k"
MD_ENFR="nmt_enfr_500k"
LG_ENFR="nmt_enfr_lg"
FR_ENFR="nmt_enfr_full"

SM_FREN="nmt_fren_50k"
MD_FREN="nmt_fren_500k"
LG_FREN="nmt_fren_lg"
FR_FREN="nmt_fren_full"

SAVENAME_SM_ENFR=$m$SM_ENFR
SAVENAME_MD_ENFR=$m$MD_ENFR
SAVENAME_LG_ENFR=$m$LG_ENFR
SAVENAME_FR_ENFR=$m$FR_ENFR

SAVENAME_SM_FREN=$m$SM_FREN
SAVENAME_MD_FREN=$m$MD_FREN
SAVENAME_LG_FREN=$m$LG_FREN
SAVENAME_FR_FREN=$m$FR_FREN


# TRANSLATE
ENTEST="../datasets/machine_translation/corpus_enfr.test.en"
FRTEST="../datasets/machine_translation/test.en.atok"
VOCAB="models/nmt_enfr_50k.pt"
BASEDIR="models/nmt_enfr_50k_base/"
ENCODER='encoder_epoch_15_accu_31.298.chkpt'
DECODER='decoder_epoch_15_accu_31.298.chkpt'
OUTPUT="outputs.txt"
EVALTXT="eval.txt"

python3 translate.py \
        -model $MODEL \
        -checkpoint_encoder $BASEDIR$ENCODER \
        -checkpoint_decoder $BASEDIR$DECODER \
        -vocab $VOCAB \
        -src $ENTEST \
        -output $BASEDIR$OUTPUT \
        -cuda

# nlg-eval --hypothesis=$BASEDIR$OUTPUT --references=$TESTDATA > $BASEDIR$EVALTXT