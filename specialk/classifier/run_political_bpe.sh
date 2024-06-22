clear
# Preprocess data
## Note: First create a dev.txt file such that
## 1. Each instance is on a new line
## 2. The label and the sentence are separated by a space

POLITICAL_DATASET="political_bpe_2019"
cnn_model_name="specialk/models/political_model_bpe_2019"
seq_len=100
# python3 specialk/classifier/preprocess_bpe.py \
#     -train_src ./datasets/political_data/classtrain.txt \
#     -label0 democratic \
#     -label1 republican \
#     -valid_src  ./datasets/political_data/classdev.txt \
#     -save_data $POLITICAL_DATASET \
#     -max_word_seq_len $seq_len \
#     -src_vocab specialk/models/nmt_fren_goldmaster_bpe_republican.pt 

# new as of 2024
# python specialk/classifier/preprocess_token.py \
#     -train_src ./datasets/political_data/classtrain.txt \
#     -label0 democratic \
#     -label1 republican \
#     -valid_src  ./datasets/political_data/classdev.txt \
#     -save_data $POLITICAL_DATASET \
#     -max_word_seq_len $seq_len \
#     -src_vocab specialk/models/nmt_fren_goldmaster_bpe_republican.pt \
#     -load_vocab \
#     -bpe


# Train the classifier (2019)
# python3 specialk/classifier/cnn_train_bpe.py \
#     -gpus 0 \
#     -epochs 1 \
#     -data $POLITICAL_DATASET".train.pt" \
#     -save_model $cnn_model_name \
#     -sequence_length $seq_len \
#     -batch_size 128 \
#     -filter_size 10 \
#     -optim adam \
#     -learning_rate 0.001 

# train the classifier (2024)
python3 specialk/classifier/cnn_train_bpe.py \
    -epochs 1 \
    -data $POLITICAL_DATASET".train.pt" \
    -save_model $cnn_model_name \
    -sequence_length $seq_len \
    -batch_size 512 \
    -filter_size 10 \
    -optim adam \
    -learning_rate 0.001 


# BESTMODEL="models/political_model_bpe_acc_98.47_loss_0.00_e1.pt"

# # # Test the classifier accuracy
# python3 cnn_translate_bpe.py \
#     -gpu 0 \
#     -model $BESTMODEL \
#     -src ../../datasets/political_data/democratic_only.test.en \
#     -tgt 'democratic' \
#     -label0 democratic \
#     -label1 republican

