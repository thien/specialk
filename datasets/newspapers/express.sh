# python3 group_publications.py

# PEN="popular.en"
# QEN="quality.en"
# TOKPATH="../tokenizer.perl"

if [ -e popular.en.atok ]
then 
    echo "Already tokenised."
else
    echo -n "performing sed.. "
    for l in en fr; do for f in *.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
    echo "done. "
    echo -n "tokenising corpus.. "
    for l in en fr
        do for f in *.$l
            do perl $TOKPATH -a -no-escape -l $l -q  < $f > $f.atok
        done
    done
    echo "done. "
fi

if [ -e popular.train.en ]
then 
    echo "Dataset already split."
else
    # split the files.
    OUTPUT_EN="popular.en"
    python3 ../splitter.py \
        -source_a $OUTPUT_EN".atok" \
        -source_b $OUTPUT_EN".atok" \
        -a_label en \
        -b_label temp \
        -verbose $true \
        -ratio 77:1:22
    OUTPUT_EN="quality.en"
    python3 ../splitter.py \
        -source_a $OUTPUT_EN".atok" \
        -source_b $OUTPUT_EN".atok" \
        -a_label en \
        -b_label temp \
        -verbose $true \
        -ratio 77:1:22
    rm *.temp
fi

CNN_TRAIN="classifier.train.en"
CNN_VALID="classifier.val.en"


if [ -e classifier.train.en ]
then
    echo "Classifier data already made."
else
    # echo $SEED > seed.txt
    cat quality.train.en > $CNN_TRAIN
    cat popular.train.en >> $CNN_TRAIN
    cat quality.val.en > $CNN_VALID
    cat popular.val.en >> $CNN_VALID
    cat $CNN_TRAIN|shuf --output=$CNN_TRAIN
    cat $CNN_VALID|shuf --output=$CNN_VALID
    # rm seed.txt
fi