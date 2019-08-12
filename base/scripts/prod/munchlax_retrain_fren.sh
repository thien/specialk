cd ..


FREN_DIRNAME="enfr_bpe_filtered_final"
MODEL="transformer"
EP=1
MODELDIM=512
BATCHSIZE=64

FORMAT="bpe"
MAXLEN="100"
PTF=".pt"
B="_base"
m="models/"
EXT="_bpe"
VOCAB_SIZE="35000"
FR_ENFR="nmt_enfr_goldmaster"$EXT
FR_FREN="nmt_fren_goldmaster"$EXT
SAVENAME_FR_ENFR=$m$FR_ENFR
SAVENAME_FR_FREN=$m$FR_FREN


python3 train.py \
    -checkpoint_encoder "models/"$FREN_DIRNAME"/encoder_epoch_1.chkpt"
    -checkpoint_decoder "models/"$FREN_DIRNAME"/decoder_epoch_1.chkpt"
    -log $true \
    -batch_size $BATCHSIZE \
    -model $MODEL \
    -epoch $EP \
    -d_word_vec $MODELDIM \
    -d_model $MODELDIM \
    -data $SAVENAME_FR_FREN$PTF \
    -save_model \
    -save_mode best \
    -directory_name $FREN_DIRNAME \
    -cuda \
    -multi_gpu


python3 core/telegram.py -m "[Munchlax] Finished second epoch of fr-en models."
