                #
# This section deals with preprocessing the political dataset
# for use when training a pre-trained NMT model.
# Note that there isn't a lot of sequences in the dataset anyway
# so it'll be fine.

cd ..

FILEPATH="./datasets/political_data/"
TRAIN_DEM=$FILEPATH"democratic_only.train.en"
TRAIN_REP=$FILEPATH"republican_only.train.en"
VALID_DEM=$FILEPATH"democratic_only.dev.en"
VALID_REP=$FILEPATH"republican_only.dev.en"
FORMAT="word"
MAXLEN="70"

PTF=".pt"
B="_base"

SM="50000"
MD="500000"

m="models/"

VOCAB_SIZE="30000"

EXT="_lower_30k_v"

# VSIZE=""
SM_REP="nmt_rep_50k"$EXT
MD_REP="nmt_rep_500k"$EXT

SM_DEM="nmt_dem_50k"$EXT
MD_DEM="nmt_dem_500k"$EXT

SAVENAME_SM_REP=$m$SM_REP
SAVENAME_MD_REP=$m$MD_REP

SAVENAME_SM_DEM=$m$SM_DEM
SAVENAME_MD_DEM=$m$MD_DEM


# ----------------
MODEL="transformer"
BEAMSIZE=3
BATCHSIZE=768

VOCAB="models/nmt_enfr_lg_lower_30k_v.pt"
ENFR_BASEDIR="models/nmt_enfr_lg_lower_30k_v_base/"
ENCODER='encoder_epoch_15_accu_42.524.chkpt'
DECODER='decoder_epoch_15_accu_42.524.chkpt'

FR=".fr"
EN=".en"


# we'll need to translate the sequences into french first.
SUBJ="../datasets/political_data/democratic_only.dev"

python3 translate.py -model $MODEL -checkpoint_encoder $ENFR_BASEDIR$ENCODER -checkpoint_decoder $ENFR_BASEDIR$DECODER -vocab $VOCAB -src $SUBJ$EN -output $SUBJ$FR -batch_size $BATCHSIZE -beam_size $BEAMSIZE -cuda

python3 core/telegram.py -m "Translated "$SUBJ$EN

SUBJ="../datasets/political_data/democratic_only.test"

python3 translate.py -model $MODEL -checkpoint_encoder $ENFR_BASEDIR$ENCODER -checkpoint_decoder $ENFR_BASEDIR$DECODER -vocab $VOCAB -src $SUBJ$EN -output $SUBJ$FR -batch_size $BATCHSIZE -beam_size $BEAMSIZE -cuda

python3 core/telegram.py -m "Translated "$SUBJ$EN

SUBJ="../datasets/political_data/democratic_only.train"

python3 translate.py -model $MODEL -checkpoint_encoder $ENFR_BASEDIR$ENCODER -checkpoint_decoder $ENFR_BASEDIR$DECODER -vocab $VOCAB -src $SUBJ$EN -output $SUBJ$FR -batch_size $BATCHSIZE -beam_size $BEAMSIZE -cuda

python3 core/telegram.py -m "Translated "$SUBJ$EN

SUBJ="../datasets/political_data/republican_only.dev"

python3 translate.py -model $MODEL -checkpoint_encoder $ENFR_BASEDIR$ENCODER -checkpoint_decoder $ENFR_BASEDIR$DECODER -vocab $VOCAB -src $SUBJ$EN -output $SUBJ$FR -batch_size $BATCHSIZE -beam_size $BEAMSIZE -cuda

python3 core/telegram.py -m "Translated "$SUBJ$EN

SUBJ="../datasets/political_data/republican_only.test"

python3 translate.py -model $MODEL -checkpoint_encoder $ENFR_BASEDIR$ENCODER -checkpoint_decoder $ENFR_BASEDIR$DECODER -vocab $VOCAB -src $SUBJ$EN -output $SUBJ$FR -batch_size $BATCHSIZE -beam_size $BEAMSIZE -cuda

python3 core/telegram.py -m "Translated "$SUBJ$EN

SUBJ="../datasets/political_data/republican_only.train"

python3 translate.py -model $MODEL -checkpoint_encoder $ENFR_BASEDIR$ENCODER -checkpoint_decoder $ENFR_BASEDIR$DECODER -vocab $VOCAB -src $SUBJ$EN -output $SUBJ$FR -batch_size $BATCHSIZE -beam_size $BEAMSIZE -cuda

python3 core/telegram.py -m "Translated "$SUBJ$EN


# # once we have the translated data then we can go ahead
# # and deal with translation back into their language.
# if [ -e ../datasets/political_data/classtrain.txt ]
# then
#     if [ -e $SAVENAME_SM_DEM$PTF ]
#     then
#         echo "small corpus already created."
#     else
#         # setup small corpus
#         python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_FR -valid_src $VALID_EN -valid_tgt $VALID_FR -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_SM_REP -max_train_seq $SM -vocab_size $VOCAB_SIZE

#         python3 preprocess.py -train_src $TRAIN_FR -train_tgt $TRAIN_EN -valid_src $VALID_FR -valid_tgt $VALID_EN -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_SM_DEM -max_train_seq $SM -vocab_size $VOCAB_SIZE
#     fi
# fi
