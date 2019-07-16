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

BEAMSIZE=1
BATCHSIZE=1024

python3 translate.py -model $MODEL -checkpoint_encoder $BASEDIR$ENCODER -checkpoint_decoder $BASEDIR$DECODER -vocab $VOCAB -src $TESTDATA -output $BASEDIR$OUTPUT -batch_size $BATCHSIZE -beam_size $BEAMSIZE -cuda

# nlg-eval --hypothesis=$BASEDIR$OUTPUT --references=$TESTDATA > $BASEDIR$EVALTXT