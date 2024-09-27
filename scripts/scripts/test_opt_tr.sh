cd ..
clear

DATA="models/nmt_ende.pt"
MODEL="transformer"
EP=10
MODELDIM=512
DIRNAME="tf_ende"

python3 optimise.py -data $DATA -log $true -model $MODEL -epoch $EP -d_word_vec $MODELDIM -d_model $MODELDIM -cuda -best_model_dir $DIRNAME