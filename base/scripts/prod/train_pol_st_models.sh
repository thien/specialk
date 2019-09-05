cd ../..

# rebase

POLDATA_DIR="../datasets/political_data/"
MODELDIR="models/fren_bpe_gold_master_mk2/"
FREN_CORP=$MODELDIR"nmt_fren_goldmaster_mk2_bpe.pt"
seq_len=100

rebase political dataset
for p in democratic republican
    do python3 rebase.py \
        -base $FREN_CORP \
        -train_src $POLDATA_DIR$p"_only.train.en.ts" \
        -train_tgt $POLDATA_DIR$p"_only.train.en" \
        -valid_src $POLDATA_DIR$p"_only.dev.en.ts" \
        -valid_tgt $POLDATA_DIR$p"_only.dev.en" \
        -save_name $MODELDIR"gm_mk2_rebase_"$p \
        -max_seq_len $(($seq_len-2))
done
python3 core/telegram.py -m "finished rebasing political datasets."


# train classifier
cd classifier

political_dataset="../"$MODELDIR"political_classifierdata"
cnn_model_name="political_bpe_classifier"

python3 preprocess_bpe.py \
    -train_src ../../datasets/political_data/classtrain.txt \
    -label0 democratic \
    -label1 republican \
    -valid_src  ../../datasets/political_data/classdev.txt \
    -save_data $political_dataset \
    -max_word_seq_len $seq_len \
    -src_vocab "../"$MODELDIR"gm_mk2_rebase_republican.pt"

# Train the classifier
python3 cnn_train_bpe.py \
    -gpus 0 \
    -epochs 1 \
    -data $political_dataset".train.pt" \
    -save_model "../"$MODELDIR$cnn_model_name \
    -sequence_length $seq_len \
    -batch_size 128 \
    -filter_size 10 \
    -optim adam \
    -learning_rate 0.001 

cd ..
# rename the model classifier
cd $MODELDIR
if [ -e $cnn_model_name".pt" ]
then
    echo "already renamed model."
else
    mv $cnn_model_name* $cnn_model_name".pt" 
fi
cd ../..
clear

# train decoder

train_st_model () {
    POLITICAL_CLASS=$1
    POLITICAL_DATA=$MODELDIR"gm_mk2_rebase_$POLITICAL_CLASS.pt"
    MODEL_FOLDERNAME="fren_bpe_gold_master_mk2"
    EP=10
    BATCHSIZE=32

    python3 train_decoder.py \
        -data $POLITICAL_DATA \
        -checkpoint_encoder $MODELDIR"/encoder_epoch_3.chkpt" \
        -checkpoint_decoder $MODELDIR"/decoder_epoch_3.chkpt" \
        -log $true \
        -batch_size $BATCHSIZE \
        -epochs $EP \
        -data $POLITICAL_DATA \
        -save_model \
        -save_mode all \
        -directory_name $MODEL_FOLDERNAME \
        -cuda \
        -multi_gpu \
        -label0 "democratic" \
        -label1 "republican" \
        -label_target $POLITICAL_CLASS \
        -classifier_model $MODELDIR"/"$cnn_model_name".pt"

    python3 core/telegram.py -m "[Casper] Finished style-transfer training for $POLITICAL_DATA"
}

train_st_model "democratic"
train_st_model "republican"

