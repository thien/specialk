

# Assumes you have the political data translated into french, and
# you have rebased it.

cd ../..


POLDATA_DIR="../datasets/political_data/"
FREN_CORP="models/nmt_fren_goldmaster_bpe.pt"
# rebase political dataset
for p in democratic republican
    do python3 rebase.py \
        -base $FREN_CORP \
        -train_src $POLDATA_DIR$p"_only.train.en.ts" \
        -train_tgt $POLDATA_DIR$p"_only.train.en" \
        -valid_src $POLDATA_DIR$p"_only.dev.en.ts" \
        -valid_tgt $POLDATA_DIR$p"_only.dev.en" \
        -save_name "models/nmt_fren_goldmaster_bpe_"$p 
done
python3 core/telegram.py -m "finished rebasing political datasets."

train_st_model () {
    POLITICAL_CLASS=$1
    POLITICAL_DATA="models/nmt_fren_goldmaster_bpe_$POLITICAL_CLASS.pt"
    POLITICAL_MODEL_OUT="fren_goldmaster_attempt_2_$POLITICAL_CLASS"
    MODEL="transformer"
    EP=1
    MODELDIM=512
    BATCHSIZE=32

    cd models
    if [ -e $POLITICAL_MODEL_OUT ]
    then
        rm -rf $POLITICAL_MODEL_OUT
    fi
    
    cp -r "fren_bpe_gold_master" $POLITICAL_MODEL_OUT
    cd ..

    python3 train.py \
        -freeze_encoder \
        -checkpoint_encoder "models/"$POLITICAL_MODEL_OUT"/encoder_epoch_2.chkpt" \
        -checkpoint_decoder "models/"$POLITICAL_MODEL_OUT"/decoder_epoch_2.chkpt" \
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

    python3 core/telegram.py -m "[Casper] Finished style-transfer training for $POLITICAL_DATA"
}

train_st_model "democratic"
train_st_model "republican"

