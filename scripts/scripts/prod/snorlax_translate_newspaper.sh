# Translate a newspaper corpus into French with a trained EN-FR model.
#
# CHECKPOINT must be a Lightning .ckpt produced by
# `python3 -m specialk.trainers.train` (see doc/generation.md).

p="popular"
DEVICE="cuda:1"
# p="quality"
# DEVICE="cuda:0"

BASEDIR="/home/t/specialk/base/"
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to a trained EN-FR checkpoint}

cd $BASEDIR

# sed "s/^[A-Za-z0-9_]*\ //" $src > $src.b

# en -> fr
PUB_DIR="../datasets/newspapers/"
BEAMSIZE=2

python3 -m specialk.translate \
    --checkpoint "$CHECKPOINT" \
    --src $PUB_DIR$p".en.atok.b" \
    --output $PUB_DIR$p".fr" \
    --beam-size $BEAMSIZE \
    --batch-size 32 \
    --max-new-tokens 150 \
    --device $DEVICE
