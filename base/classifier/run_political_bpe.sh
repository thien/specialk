clear
# Preprocess data
## Note: First create a dev.txt file such that
## 1. Each instance is on a new line
## 2. The label and the sentence are separated by a space

political_dataset="political_bpe"
cnn_model_name="models/political_model_bpe"
seq_len=100
# python3 preprocess_bpe.py \
#     -train_src ../../datasets/political_data/classtrain.txt \
#     -label0 democratic \
#     -label1 republican \
#     -valid_src  ../../datasets/political_data/classdev.txt \
#     -save_data $political_dataset \
#     -max_word_seq_len $seq_len \
#     -src_vocab ../models/nmt_fren_goldmaster_bpe_republican.pt 


# Train the classifier
python3 cnn_train_bpe.py \
    -gpus 0 \
    -epochs 20 \
    -data $political_dataset".train.pt" \
    -save_model $cnn_model_name \
    -sequence_length $seq_len \
    -batch_size 128 \
    -filter_size 10 \
    -optim adam \
    -learning_rate 0.001 

# BESTMODEL="political_model.pt"

# # Test the classifier accuracy
# python3 cnn_translate.py \
#     -gpu 0 \
#     -model $BESTMODEL \
#     -src ../../datasets/political_data/democratic_only.test.en \
#     -tgt 'democratic' \
#     -label0 democratic \
#     -label1 republican

