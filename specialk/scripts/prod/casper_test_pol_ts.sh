

# transformer variables

style_transfer () {
    pol_input=$1
    pol_output=$2
    model_num=$3

    encoder="models/fren_goldmaster_"$pol_output"/encoder_epoch_"$model_num".chkpt"
    decoder="models/fren_goldmaster_"$pol_output"/decoder_epoch_"$model_num".chkpt"
    vocab="models/nmt_fren_goldmaster_bpe_"$pol_output".pt"
    source="../datasets/political_data/"$pol_input"_only.test.en.ts"
    target="results/pol/"$pol_input"_only.test.to."$pol_output"."$model_num

    
    # back translate
    python3 translate.py \
        -model transformer \
        -checkpoint_encoder $encoder \
        -checkpoint_decoder $decoder \
        -vocab $vocab \
        -src $source \
        -output $target \
        -beam_size 1 \
        -cuda
    
    # # classify output
        
    BESTMODEL="classifier/political_model.pt"

    # Test the classifier accuracy
    python3 classifier/cnn_translate.py \
    -gpu 0 \
    -model $BESTMODEL \
    -src $target \
    -tgt $pol_output \
    -label0 democratic \
    -label1 republican

}
cd ../..
# best democratic: 6
# best republican 7

textout="results/pol/out_attempt_new_"
style_transfer "democratic" "republican" 1 > $textout$n".dr.txt" 
style_transfer "republican" "democratic" 1 > $textout$n".rd.txt"
