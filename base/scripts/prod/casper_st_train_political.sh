

# Assumes you have the political data translated into french, and
# you have rebased it.

cd ../..

POLITICAL_CLASS="democratic"

POLITICAL_DATA="models/nmt_fren_goldmaster_bpe_$POLITICAL_CLASS.pt"
POLITICAL_MODEL_OUT="nmt_fren_goldmaster_$POLITICAL_CLASS"
FREN_DIRNAME="models/enfr_bpe_gold_master"
MODEL="transformer"
EP=5
MODELDIM=512
BATCHSIZE=32

python3 train.py \
    -checkpoint_encoder $FREN_DIRNAME"/encoder_epoch_1.chkpt" \
    -checkpoint_decoder $FREN_DIRNAME"/decoder_epoch_1.chkpt" \
    -log $true \
    -batch_size $BATCHSIZE \
    -model $MODEL \
    -epoch $EP \
    -d_word_vec $MODELDIM \
    -d_model $MODELDIM \
    -data $POLITICAL_DATA \
    -save_model \
    -save_mode all \
    -directory_name $POLITICAL_MODEL_OUT \
    -cuda \
    -multi_gpu

python3 core/telegram.py -m "[Munchlax] Finished style-transfer training for $POLITICAL_DATA"
