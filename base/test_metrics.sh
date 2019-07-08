clear
DOCA="../datasets/multi30k/test.de.atok"
ALANG="de"
DOCB="models/transformer-19-06-26-18-12-36/outputs.txt"
BLANG="de"
python3 metrics.py -doc_a $DOCA -doc_a_lang $ALANG -doc_b $DOCB -doc_b_lang $BLANG -bleu