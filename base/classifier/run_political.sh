
# Preprocess data
## Note: First create a dev.txt file such that
## 1. Each instance is on a new line
## 2. The label and the sentence are separated by a space

political_dataset="political_new"
cnn_model_name="political_model_new"
python3 preprocess_token.py \
    -train_src ../../datasets/political_data/classtrain.txt \
    -label0 democratic \
    -label1 republican \
    -valid_src  ../../datasets/political_data/classdev.txt \
    -save_data $political_dataset \
    -src_vocab ../results/nmt_fren_goldmaster_bpe_republican.tgt.dict \
    -seq_length 80 \

# Train the classifier
# python3 cnn_train.py \
#     -gpus 0 \
#     -epochs 20 \
#     -data $political_dataset".train.pt" \
#     -save_model $cnn_model_name \
#     -batch_size 128
     # -sequence_length 100 \

# BESTMODEL="political_model.pt"

# # Test the classifier accuracy
# python3 cnn_translate.py \
#     -gpu 0 \
#     -model $BESTMODEL \
#     -src ../../datasets/political_data/democratic_only.test.en \
#     -tgt 'democratic' \
#     -label0 democratic \
#     -label1 republican

