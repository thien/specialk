

# transformer variables

style_transfer () {
    pol_input=$1
    pol_output=$2
    model_num=$3
    checkpoint=$4

    source="../datasets/political_data/"$pol_input"_only.test.en.ts"
    target="results/pol/"$pol_input"_only.test.to."$pol_output"."$model_num


    # back translate
    python3 -m specialk.translate \
        --checkpoint "$checkpoint" \
        --src "$source" \
        --output "$target" \
        --beam-size 1

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

# One trained style-transfer checkpoint per direction, from
# `python3 -m specialk.trainers.train --dataset political`.
CHECKPOINT_DEMOCRATIC_TO_REPUBLICAN=${CHECKPOINT_DEMOCRATIC_TO_REPUBLICAN:?set CHECKPOINT_DEMOCRATIC_TO_REPUBLICAN}
CHECKPOINT_REPUBLICAN_TO_DEMOCRATIC=${CHECKPOINT_REPUBLICAN_TO_DEMOCRATIC:?set CHECKPOINT_REPUBLICAN_TO_DEMOCRATIC}

textout="results/pol/out_attempt_new_"
style_transfer "democratic" "republican" 1 "$CHECKPOINT_DEMOCRATIC_TO_REPUBLICAN" > $textout$n".dr.txt"
style_transfer "republican" "democratic" 1 "$CHECKPOINT_REPUBLICAN_TO_DEMOCRATIC" > $textout$n".rd.txt"
