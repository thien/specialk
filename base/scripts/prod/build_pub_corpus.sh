SRCDIR="/home/t/project_model/datasets/"
cd $SRCDIR

#
# TOKENISING
#

if [ -e $SRCDIR"/newspapers/ready/sacrebleu_quality_filt.fr.atok" ]
then
    echo "already tokenised"
else
    echo -n "tokenising corpus.. "
    for l in fr
        do for f in newspapers/ready/*.$l
            do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok
        done
    done
fi


for y in popular quality
do 
    OUTPUT_EN="newspapers/ready/sacrebleu_"$y"_filt.en"
    OUTPUT_FR="newspapers/ready/sacrebleu_"$y"_filt.fr.atok"
    echo "splitting dataset into training, validation and test data."
    python3 splitter.py \
       -source_a $OUTPUT_EN \
       -source_b $OUTPUT_FR \
       -a_label en \
       -b_label fr \
       -verbose $true \
       -ratio 97:0.1:2.9
done
