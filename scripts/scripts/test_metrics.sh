clear
cd ..

GLOVE="/media/data/Datasets/glove/glove.6B.50d.txt"
DOCA="../datasets/multi30k/test.de.atok"
ALANG="de"
DOCB="models/transformer-19-06-26-18-12-36/outputs.txt"
BLANG="de"
OUT="models/transformer-19-06-26-18-12-36/stats.json"
python3 metrics.py \
        -reference $DOCA \
        -ref_lang $ALANG \
        -hypothesis $DOCB \
        -hyp_lang $BLANG \
        -output $OUT \
        -glove_path $GLOVE \
        -bleu \
        -rouge \
        -wmd \
        -meteor

# TODO: make meteor measurements
# TODO: add perplexity(??)

# TODO: lexical matches
# TODO: 