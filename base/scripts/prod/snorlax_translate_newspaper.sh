# assumes you transferred the GM models from ./snorlax_to_casper.sh

p="popular"
CUDA_DEVICE=1
# p="quality"
# CUDA_DEVICE=0

BASEDIR="/home/t/specialk/base/"
MODELDIR=$BASEDIR"models/"

cd $BASEDIR

ENFR_DIRNAME="enfr_bpe_gold_master"
FREN_DIRNAME="fren_bpe_gold_master"
VOCAB=$MODELDIR"nmt_enfr_goldmaster_bpe.pt"
# translate political dataset
ENFR_BASEDIR="models/"$ENFR_DIRNAME"/"
ENCODER='encoder_epoch_1.chkpt'
DECODER='decoder_epoch_1.chkpt'
OUTPUT="outputs.txt"

# en -> fr
PUB_DIR="../datasets/newspapers/"


python3 translate.py \
    -model transformer \
    -checkpoint_encoder $ENFR_BASEDIR$ENCODER \
    -checkpoint_decoder $ENFR_BASEDIR$DECODER \
    -vocab $VOCAB \
    -src $PUB_DIR$p".en.atok" \
    -output $PUB_DIR$p".fr" \
    -cuda_device $CUDA_DEVICE \
    -cuda
python3 core/telegram.py -m "finished translating $src"
