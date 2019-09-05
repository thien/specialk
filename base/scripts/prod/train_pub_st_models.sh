# rebase dataset

BASEDIR="/home/t/project_model/base"
MODEL_FOLDERNAME="fren_bpe_gm_mk2_articles_mk1"
MODELDIR=$BASEDIR"/models/"$MODEL_FOLDERNAME

LABEL0="popular"
LABEL1="quality"
BASE_FREN_ENCODER=$BASEDIR"/models/fren_bpe_gold_master_mk2/encoder_epoch_3.chkpt"
BASE_FREN_DECODER=$BASEDIR"/models/fren_bpe_gold_master_mk2/decoder_epoch_3.chkpt"
# VOCABSRC is the vocabulary you want to use to rebase your dataset on.
VOCABSRC=$BASEDIR"/models/fren_bpe_gold_master_mk2/gm_mk2_rebase_democratic.pt"

SEQ_LEN=100
DATASET_SRC="/home/t/project_model/datasets/newspapers/ready"

# DATASET_EN_TRAIN=$DATASET_SRC"/sacrebleu_"$p"_filt.train.en"
# DATASET_EN_VALID=$DATASET_SRC"/sacrebleu_"$p"_filt.val.en"

# DATASET_FR_TRAIN=$DATASET_SRC"/sacrebleu_"$p"_filt.train.fr"
# DATASET_FR_VALID=$DATASET_SRC"/sacrebleu_"$p"_filt.val.fr"

# DATASET_FR_TEST=$DATASET_SRC"/sacrebleu_"$p"_filt.test.fr"
# DATASET_EN_TEST=$DATASET_SRC"/sacrebleu_"$p"_filt.test.en"

VOCABTGTDIR=$MODELDIR


CNN_DATASET=$MODELDIR"/articles_classifierdata"
CNN_MODEL_FILEPATH=$MODELDIR"/bpe_classifier"

DATASET_TAG="/gm_mk2_rebase_"

# ------------------------------------------------------------------------------

mkdir -p $MODELDIR"/st_results/stat"

cd $BASEDIR

# rebase dataset
for p in quality popular
    do if [ -e $VOCABTGTDIR$DATASET_TAG$p".pt" ]
    then
        echo "already rebased "$p" dataset."
    else
        python3 rebase.py \
            -base $VOCABSRC \
            -train_src $DATASET_SRC"/sacrebleu_"$p"_filt.train.fr" \
            -train_tgt $DATASET_SRC"/sacrebleu_"$p"_filt.train.en" \
            -valid_src $DATASET_SRC"/sacrebleu_"$p"_filt.val.fr" \
            -valid_tgt $DATASET_SRC"/sacrebleu_"$p"_filt.val.en" \
            -save_name $VOCABTGTDIR$DATASET_TAG$p \
            -max_seq_len $(($SEQ_LEN-2))
    fi
done

# create classifier data
mkdir -p $DATASET_SRC"/classifier"

CLASSIFIER_TRAIN_DATA=$DATASET_SRC"/classifier/train.txt"
CLASSIFIER_VALID_DATA=$DATASET_SRC"/classifier/val.txt"

if [ -e $CLASSIFIER_TRAIN_DATA ]
then
    echo "already made classifier data"
else
    sed -e 's/^/quality /' $DATASET_SRC"/sacrebleu_quality_filt.train.en" > $CLASSIFIER_TRAIN_DATA
    sed -e 's/^/popular /' $DATASET_SRC"/sacrebleu_popular_filt.train.en" >> $CLASSIFIER_TRAIN_DATA
    # shuffle classifier dataset
    shuf $CLASSIFIER_TRAIN_DATA --output=$CLASSIFIER_TRAIN_DATA
    sed -e 's/^/quality /' $DATASET_SRC"/sacrebleu_quality_filt.val.en" > $CLASSIFIER_VALID_DATA
    sed -e 's/^/popular /' $DATASET_SRC"/sacrebleu_popular_filt.val.en" >> $CLASSIFIER_VALID_DATA
    # shuffle classifier dataset
    shuf $CLASSIFIER_VALID_DATA --output=$CLASSIFIER_VALID_DATA
fi

cd $BASEDIR"/classifier"

if [ -e $CNN_DATASET".train.pt" ]
then
    echo "already made" $CNN_DATASET".train.pt" 
else
    python3 preprocess_bpe.py \
        -train_src $CLASSIFIER_TRAIN_DATA \
        -label0 $LABEL0 \
        -label1 $LABEL1 \
        -valid_src  $CLASSIFIER_VALID_DATA \
        -save_data $CNN_DATASET \
        -src_vocab $VOCABSRC \
        -max_word_seq_len $SEQ_LEN
fi

if [ -e $CNN_MODEL_FILEPATH".pt" ]
then
    echo "already made classifier"
else
    # Train the classifier
    python3 cnn_train_bpe.py \
        -gpus 0 \
        -epochs 1 \
        -data $CNN_DATASET".train.pt" \
        -save_model $CNN_MODEL_FILEPATH \
        -sequence_length $SEQ_LEN \
        -batch_size 128 \
        -filter_size 5 \
        -optim adam \
        -learning_rate 0.001 
    # rename the model classifier
    mv $CNN_MODEL_FILEPATH* $CNN_MODEL_FILEPATH".pt" 
fi

cd $BASEDIR

echo "Setting up Style Transfer Model Training"

# train decoder for each of the style models
train_st_model () {
    PUBLICATION_CLASS=$1
    PUBLICATION_DATA=$MODELDIR$DATASET_TAG$PUBLICATION_CLASS".pt"
    EP=5
    BATCHSIZE=32

    python3 train_decoder.py \
        -data $PUBLICATION_DATA \
        -checkpoint_encoder $BASE_FREN_ENCODER \
        -checkpoint_decoder $BASE_FREN_DECODER \
        -log \
        -batch_size $BATCHSIZE \
        -epochs $EP \
        -save_model \
        -save_mode all \
        -directory_name $MODEL_FOLDERNAME \
        -cuda \
        -new_directory \
        -multi_gpu \
        -label0 $LABEL0 \
        -label1 $LABEL1 \
        -label_target $PUBLICATION_CLASS \
        -classifier_model $CNN_MODEL_FILEPATH".pt"

    python3 core/telegram.py -m "[Casper] Finished style-transfer training for $PUBLICATION_DATA"
}

#train_st_model $LABEL0
#train_st_model $LABEL1

# # test models
style_transfer () {
    pol_input=$1
    pol_output=$2
    model_num=$3

    decoder=$MODELDIR"/decoder_"$pol_output"_epoch_"$model_num".chkpt"
    vocab=$MODELDIR$DATASET_TAG$pol_output".pt"
    source=$DATASET_SRC"/sacrebleu_"$pol_input"_filt.test.fr"
    target=$MODELDIR"/st_results/"$pol_input"_only.test.to."$pol_output"."$model_num

    cd $BASEDIR
    
#     back translate
    python3 translate.py \
        -model transformer \
        -checkpoint_encoder $BASE_FREN_ENCODER \
        -checkpoint_decoder $decoder \
        -vocab $vocab \
        -src $source \
        -output $target \
        -beam_size 5 \
        -batch_size 10 \
        -cuda
    
    # classify output

    cd classifier
    # Test the classifier accuracy
    python3 cnn_translate_bpe.py \
    -gpu 0 \
    -model $CNN_MODEL_FILEPATH".pt" \
    -src $target \
    -tgt $pol_output \
    -label0 $LABEL0 \
    -label1 $LABEL1
    
    cd $BASEDIR
}

# # measure performance

textout=$MODELDIR"/st_results/stat/"

for x in 1 2 3 4 5 
do 
#    style_transfer $LABEL0 $LABEL1 $x > $textout"01."$x".txt"
    style_transfer $LABEL1 $LABEL0 $x > $textout"10."$x".txt"
done
