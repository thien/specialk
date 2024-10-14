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

VOCABTGTDIR=$MODELDIR


CNN_DATASET=$MODELDIR"/articles_classifierdata"
CNN_MODEL_FILEPATH=$MODELDIR"/bpe_classifier"

DATASET_TAG="/gm_mk2_rebase_"

cd $BASEDIR


NATURALNESSDIR=$MODELDIR"/naturalness_test"

# make the naturalness directory (if it doesnt exist)
mkdir -p $NATURALNESSDIR

# make a directory for your real and fake data
gen_fake_seq () {
    pol_input=$1

    decoder=$MODELDIR"/decoder_"$pol_input"_epoch_3.chkpt"
    vocab=$MODELDIR$DATASET_TAG$pol_input".pt"

    DATASET_EN_TRAIN=$DATASET_SRC"/sacrebleu_"$pol_input"_filt.train.en"
    DATASET_EN_VALID=$DATASET_SRC"/sacrebleu_"$pol_input"_filt.val.en"

    DATASET_FR_TRAIN=$DATASET_SRC"/sacrebleu_"$pol_input"_filt.train.fr"
    DATASET_FR_VALID=$DATASET_SRC"/sacrebleu_"$pol_input"_filt.val.fr"

    # python3 translate.py \
    #     -model transformer \
    #     -checkpoint_encoder $BASE_FREN_ENCODER \
    #     -checkpoint_decoder $decoder \
    #     -vocab $vocab \
    #     -src $DATASET_FR_TRAIN \
    #     -output $NATURALNESSDIR"/"$pol_input+".train.en.fake" \
    #     -beam_size 5 \
    #     -batch_size 10 \
    #     -cuda

    python3 translate.py \
        -model transformer \
        -checkpoint_encoder $BASE_FREN_ENCODER \
        -checkpoint_decoder $decoder \
        -vocab $vocab \
        -src $DATASET_FR_VALID \
        -output $NATURALNESSDIR"/"$pol_input+".val.en.fake" \
        -beam_size 5 \
        -batch_size 10 \
        -cuda

    # cp $DATASET_EN_TRAIN $NATURALNESSDIR"/"$pol_input+".train.en.real"
    cp $DATASET_EN_VALID $NATURALNESSDIR"/"$pol_input+".val.en.real"
}

# gen_fake_seq $LABEL0
# gen_fake_seq $LABEL1


gen_fake_classify () {
    pol_input=$1
    pol_output=$2
    # echo $pol_input ">" $pol_output
    before_src=$DATASET_SRC"/sacrebleu_"$pol_input"_filt.test.en"
    st_src=$MODELDIR"/st_results/"$pol_input"_only.test.to."$pol_output"."3
    cd $BASEDIR

    # measure preservation of meaning and naturalness
    # python3 measure_ts.py -src $before_src -tgt $before_src -type publication 

    # newspaper specific stuff
    python3 measurements.py -reference $before_src -ref_lang en -hypothesis $before_src -output $MODELDIR"/qual_nochange.stat"
  
    # cd $BASEDIR"/classifier"
    # python3 cnn_translate_bpe.py \
    #     -gpu 0 \
    #     -model $MODELDIR"/bpe_classifier.pt" \
    #     -src $before_src \
    #     -tgt $pol_output \
    #     -label0 $LABEL0 \
    #     -label1 $LABEL1
}
# gen_fake_classify $LABEL1 $LABEL0

NATURALNESS_TRAINDATA=$NATURALNESSDIR"/naturalness.train"
NATURALNESS_VALIDDATA=$NATURALNESSDIR"/naturalness.valid"
LABEL0="fake"
LABEL1="real"

# sed -e 's/^/real /' $NATURALNESSDIR"/real.train" > $NATURALNESS_TRAINDATA
# sed -e 's/^/fake /' $NATURALNESSDIR"/fake.train" >> $NATURALNESS_TRAINDATA
# # shuffle classifier dataset
# shuf $NATURALNESS_TRAINDATA --output=$NATURALNESS_TRAINDATA
# sed -e 's/^/real /' $NATURALNESSDIR"/real.val" > $NATURALNESS_VALIDDATA
# sed -e 's/^/fake /' $NATURALNESSDIR"/fake.val" >> $NATURALNESS_VALIDDATA
# # shuffle classifier dataset
# shuf $NATURALNESS_VALIDDATA --output=$NATURALNESS_VALIDDATA


# need to translate training and validation sequences
# for quality and popular paper

CNN_MODEL_NATURALNESS_FILEPATH=$NATURALNESSDIR"/naturalness_classifier"
CNN_NATURALNESS_DATSET=$NATURALNESSDIR"/naturalness_dataset"

pop_to_qual_test=$MODELDIR"/st_results/popular_only.test.to.quality.3"
qual_to_pop_test=$MODELDIR"/st_results/quality_only.test.to.popular.3"

cd $BASEDIR"/classifier"

# python3 preprocess_token.py \
#     -train_src $NATURALNESS_TRAINDATA \
#     -label0 fake \
#     -label1 real \
#     -valid_src  $NATURALNESS_VALIDDATA \
#     -save_data $CNN_NATURALNESS_DATSET \
#     -src_vocab_size 40000 \
#     # -src_vocab ../results/nmt_fren_goldmaster_bpe_republican.tgt.dict \
#     -seq_length 50

# python3 cnn_train.py \
#     -gpus 0 \
#     -epochs 2 \
#     -data $CNN_NATURALNESS_DATSET".train.pt" \
#     -save_model $CNN_MODEL_NATURALNESS_FILEPATH \
#     -batch_size 128 \
#     -sequence_length 50


echo $pop_to_qual_test
python3 cnn_translate.py \
    -gpu 0 \
    -model $NATURALNESSDIR"/naturalness_classifier_acc_70.72_loss_0.00_e2.pt"\
    -src $pop_to_qual_test \
    -tgt "real" \
    -label0 $LABEL0 \
    -label1 $LABEL1

echo $qual_to_pop_test
python3 cnn_translate.py \
    -gpu 0 \
    -model $NATURALNESSDIR"/naturalness_classifier_acc_70.72_loss_0.00_e2.pt"\
    -src $qual_to_pop_test \
    -tgt "real" \
    -label0 $LABEL0 \
    -label1 $LABEL1


if [ -e $CNN_NATURALNESS_DATSET".train.pt" ]
then
    echo "already made" $CNN_NATURALNESS_DATSET".train.pt" 
else
    python3 preprocess_bpe.py \
        -train_src $NATURALNESS_TRAINDATA \
        -label0 $LABEL0 \
        -label1 $LABEL1 \
        -valid_src $NATURALNESS_VALIDDATA \
        -save_data $CNN_NATURALNESS_DATSET \
        -src_vocab $VOCABSRC \
        -max_word_seq_len $SEQ_LEN
fi

if [ -e $CNN_MODEL_NATURALNESS_FILEPATH".pt" ]
then
    echo "already made classifier"
else
    # Train the classifier
    python3 cnn_train_bpe.py \
        -gpus 0 \
        -epochs 1 \
        -data $CNN_NATURALNESS_DATSET".train.pt" \
        -save_model $CNN_MODEL_NATURALNESS_FILEPATH \
        -sequence_length $SEQ_LEN \
        -batch_size 128 \
        -filter_size 4 \
        -optim adam \
        -learning_rate 0.001 
    # rename the model classifier
    mv $CNN_MODEL_NATURALNESS_FILEPATH* $CNN_MODEL_NATURALNESS_FILEPATH".pt" 
fi

# echo $pop_to_qual_test
# python3 cnn_translate.py \
#     -gpu 0 \
#     -model $NATURALNESSDIR"/naturalness_classifier_acc_70.72_loss_0.00_e2.pt"\
#     -src $pop_to_qual_test \
#     -tgt "real" \
#     -label0 $LABEL0 \
#     -label1 $LABEL1

# echo $qual_to_pop_test
# python3 cnn_translate.py \
#     -gpu 0 \
#     -model $NATURALNESSDIR"/naturalness_classifier_acc_70.72_loss_0.00_e2.pt"\
#     -src $qual_to_pop_test \
#     -tgt "real" \
#     -label0 $LABEL0 \
#     -label1 $LABEL1