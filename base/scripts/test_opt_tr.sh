cd ..
clear

DATA="models/nmt_ende.pt"
MODEL="transformer"
EP=1
MODELDIM=512

python3 optimise.py -data $DATA -log $true -model $MODEL -epoch $EP -d_word_vec $MODELDIM -d_model $MODELDIM -cuda