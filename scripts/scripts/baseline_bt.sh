#!/bin/bash 
# wizard script to create optimal en-fr and fr-en NMT Transformer model.

MODEL="transformer"
BEAMSIZE=3
BATCHSIZE=768
# TESTINP_EN="../datasets/machine_translation/corpus_enfr.test.en"
# TESTINP_FR="../datasets/machine_translation/corpus_enfr.test.fr"
TESTINP_EN="../datasets/machine_translation/corpus_enfr.test.en.sm"
TESTINP_FR="../datasets/machine_translation/corpus_enfr.test.fr.sm"
METRICSDIR="metrics/baseline_bt/"

cd ..

#
# TRANSLATE SM
#

#en->fr

VOCAB="models/nmt_enfr_50k_lower_30k_v.pt"
ENFR_BASEDIR="models/nmt_enfr_50k_lower_30k_v_base/"
ENCODER='encoder_epoch_13.chkpt'
DECODER='decoder_epoch_13.chkpt'

ENFR_OUTPUT="outputs_50k_lower_enfr.txt"
EVALTXT=$METRICSDIR"eval_50k_lower_enfr.txt"

# python3 translate.py -model $MODEL -checkpoint_encoder $ENFR_BASEDIR$ENCODER -checkpoint_decoder $ENFR_BASEDIR$DECODER -vocab $VOCAB -src $TESTINP_EN -output $ENFR_BASEDIR$ENFR_OUTPUT -batch_size $BATCHSIZE -beam_size $BEAMSIZE -cuda

# nlg-eval --hypothesis=$ENFR_BASEDIR$ENFR_OUTPUT --references=$TESTINP_FR > $EVALTXT

#fr->en

VOCAB="models/nmt_fren_50k_lower_30k_v.pt"
FREN_BASEDIR="models/nmt_fren_50k_lower_30k_v_base/"
ENCODER='encoder_epoch_15.chkpt'
DECODER='decoder_epoch_15.chkpt'

FREN_OUTPUT="outputs_50k_lower_fren.txt"
EVALTXT=$METRICSDIR"eval_50k_lower_fren.txt"

# python3 translate.py -model $MODEL -checkpoint_encoder $FREN_BASEDIR$ENCODER -checkpoint_decoder $FREN_BASEDIR$DECODER -vocab $VOCAB -src $ENFR_BASEDIR$ENFR_OUTPUT -output $FREN_BASEDIR$FREN_OUTPUT -batch_size $BATCHSIZE -beam_size $BEAMSIZE -cuda

# nlg-eval --hypothesis=$FREN_BASEDIR$FREN_OUTPUT --references=$TESTINP_EN > $EVALTXT



#
# TRANSLATE MD
#

#en->fr

VOCAB="models/nmt_enfr_500k_lower_30k_v.pt"
ENFR_BASEDIR="models/nmt_enfr_500k_lower_30k_v_base/"
ENCODER='encoder_epoch_15_accu_39.029.chkpt'
DECODER='decoder_epoch_15_accu_39.029.chkpt'

ENFR_OUTPUT="outputs_500k_lower_enfr.txt"
EVALTXT=$METRICSDIR"eval_500k_lower_enfr.txt"

# python3 translate.py -model $MODEL -checkpoint_encoder $ENFR_BASEDIR$ENCODER -checkpoint_decoder $ENFR_BASEDIR$DECODER -vocab $VOCAB -src $TESTINP_EN -output $ENFR_BASEDIR$ENFR_OUTPUT -batch_size $BATCHSIZE -beam_size $BEAMSIZE -cuda

# nlg-eval --hypothesis=$ENFR_BASEDIR$ENFR_OUTPUT --references=$TESTINP_FR > $EVALTXT

#fr->en

VOCAB="models/nmt_fren_500k_lower_30k_v.pt"
FREN_BASEDIR="models/nmt_fren_500k_lower_30k_v_base/"
ENCODER='encoder_epoch_15_accu_37.859.chkpt'
DECODER='decoder_epoch_15_accu_37.859.chkpt'

FREN_OUTPUT="outputs_500k_lower_fren.txt"
EVALTXT=$METRICSDIR"eval_500k_lower_fren.txt"

# python3 translate.py -model $MODEL -checkpoint_encoder $FREN_BASEDIR$ENCODER -checkpoint_decoder $FREN_BASEDIR$DECODER -vocab $VOCAB -src $ENFR_BASEDIR$ENFR_OUTPUT -output $FREN_BASEDIR$FREN_OUTPUT -batch_size $BATCHSIZE -beam_size $BEAMSIZE -cuda

# nlg-eval --hypothesis=$FREN_BASEDIR$FREN_OUTPUT --references=$TESTINP_EN > $EVALTXT



#
# TRANSLATE LG
#

#en->fr

VOCAB="models/nmt_enfr_lg_lower_30k_v.pt"
ENFR_BASEDIR="models/nmt_enfr_lg_lower_30k_v_base/"
ENCODER='encoder_epoch_15_accu_42.524.chkpt'
DECODER='decoder_epoch_15_accu_42.524.chkpt'

ENFR_OUTPUT="outputs_lg_lower_enfr.txt"
EVALTXT=$METRICSDIR"eval_lg_lower_enfr.txt"

# python3 translate.py -model $MODEL -checkpoint_encoder $ENFR_BASEDIR$ENCODER -checkpoint_decoder $ENFR_BASEDIR$DECODER -vocab $VOCAB -src $TESTINP_EN -output $ENFR_BASEDIR$ENFR_OUTPUT -batch_size $BATCHSIZE -beam_size $BEAMSIZE -cuda

python3 core/telegram.py -m "Initiating measurements."

nlg-eval --hypothesis=$ENFR_BASEDIR$ENFR_OUTPUT --references=$TESTINP_FR > $EVALTXT

python3 core/telegram.py -m "Finished nlg-eval for lg en->fr"
#fr->en

VOCAB="models/nmt_fren_lg_lower_30k_v.pt"
FREN_BASEDIR="models/nmt_fren_lg_lower_30k_v_base/"
ENCODER='encoder_epoch_15_accu_41.363.chkpt'
DECODER='decoder_epoch_15_accu_41.363.chkpt'

FREN_OUTPUT="outputs_lg_lower_fren.txt"
EVALTXT=$METRICSDIR"eval_lg_lower_fren.txt"

# python3 translate.py -model $MODEL -checkpoint_encoder $FREN_BASEDIR$ENCODER -checkpoint_decoder $FREN_BASEDIR$DECODER -vocab $VOCAB -src $ENFR_BASEDIR$ENFR_OUTPUT -output $FREN_BASEDIR$FREN_OUTPUT -batch_size $BATCHSIZE -beam_size $BEAMSIZE -cuda

nlg-eval --hypothesis=$FREN_BASEDIR$FREN_OUTPUT --references=$TESTINP_EN > $EVALTXT


python3 core/telegram.py -m "finished stats."
