# wizard script to create optimal en-fr and fr-en NMT Transformer model.

cd ..

# TRANSLATE
MODEL="transformer"
TESTDATA="../datasets/machine_translation/corpus_enfr.test.en"
VOCAB="models/nmt_enfr_50k_lower_30k_v.pt"
BASEDIR="models/nmt_enfr_50k_lower_30k_v_base/"
ENCODER='encoder_epoch_13.chkpt'
DECODER='decoder_epoch_13.chkpt'
# VOCAB="models/nmt_enfr_full_lower_30k_v.pt"
# BASEDIR="models/nmt_enfr_lg_lower_30k_v_base/"
# ENCODER='encoder_epoch_15_accu_42.524.chkpt'
# DECODER='decoder_epoch_15_accu_42.524.chkpt'
OUTPUT="outputs.txt"
EVALTXT="eval.txt"

BEAMSIZE=3
BATCHSIZE=1024

# python3 translate.py -model $MODEL -checkpoint_encoder $BASEDIR$ENCODER -checkpoint_decoder $BASEDIR$DECODER -vocab $VOCAB -src $TESTDATA -output $BASEDIR$OUTPUT -batch_size $BATCHSIZE -beam_size $BEAMSIZE -cuda

# nlg-eval --hypothesis=$BASEDIR$OUTPUT --reference s=$TESTDATA > $BASEDIR$EVALTXT

MODEL="transformer"
FILEPATH="../datasets/machine_translation/"
TRAIN_EN=$FILEPATH"corpus_enfr.train.en"
TRAIN_FR=$FILEPATH"corpus_enfr.train.fr"
VALID_EN=$FILEPATH"corpus_enfr.val.en"
VALID_FR=$FILEPATH"corpus_enfr.val.fr"
FORMAT="word"
MAXLEN="70"

PTF=".pt"
B="_base"

SM="50000"
MD="500000"
LG="1500000"

m="models/"

VOCAB_SIZE="30000"

EXT="_lower_30k_v"

# VSIZE=""
SM_ENFR="nmt_enfr_50k"$EXT
MD_ENFR="nmt_enfr_500k"$EXT
LG_ENFR="nmt_enfr_lg"$EXT
FR_ENFR="nmt_enfr_full"$EXT

SM_FREN="nmt_fren_50k"$EXT
MD_FREN="nmt_fren_500k"$EXT
LG_FREN="nmt_fren_lg"$EXT
FR_FREN="nmt_fren_full"$EXT

SAVENAME_SM_ENFR=$m$SM_ENFR
SAVENAME_MD_ENFR=$m$MD_ENFR
SAVENAME_LG_ENFR=$m$LG_ENFR
SAVENAME_FR_ENFR=$m$FR_ENFR

SAVENAME_SM_FREN=$m$SM_FREN
SAVENAME_MD_FREN=$m$MD_FREN
SAVENAME_LG_FREN=$m$LG_FREN
SAVENAME_FR_FREN=$m$FR_FREN

# 50k

# 500k

# lg

echo $SAVENAME_SM_ENFR
echo $SAVENAME_MD_ENFR
echo $SAVENAME_LG_ENFR
echo $SAVENAME_SM_FREN
echo $SAVENAME_MD_FREN
echo $SAVENAME_LG_FREN