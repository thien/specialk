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

# sed "s/^[A-Za-z0-9_]*\ //" $src > $src.b

# en -> fr
PUB_DIR="../datasets/newspapers/"
BEAMSIZE=2

python3 translate.py \
    -model transformer \
    -checkpoint_encoder $ENFR_BASEDIR$ENCODER \
    -checkpoint_decoder $ENFR_BASEDIR$DECODER \
    -vocab $VOCAB \
    -src $PUB_DIR$p".en.atok.b" \
    -output $PUB_DIR$p".fr" \
    -beam_size $BEAMSIZE \
    -batch_size 32 \
    -override_max_token_seq_len 150 \
    -cuda_device $CUDA_DEVICE \
    -cuda
python3 core/telegram.py -m "finished translating $src"
