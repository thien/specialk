clear
DOCA="../datasets/multi30k/test.de.atok"
ALANG="de"
DOCB="models/transformer-19-06-26-18-12-36/outputs.txt"
BLANG="de"
python3 metrics.py -reference $DOCA -ref_lang $ALANG -hypothesis $DOCB -hyp_lang $BLANG -bleu -rouge