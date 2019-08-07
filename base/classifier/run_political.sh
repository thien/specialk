
# Preprocess data
## Note: First create a dev.txt file such that
## 1. Each instance is on a new line
## 2. The label and the sentence are separated by a space
# python3 preprocess.py \
#     -train_src ../../datasets/political_data/classtrain.txt \
#     -label0 democratic \
#     -label1 republican \
#     -valid_src  ../../datasets/political_data/classdev.txt \
#     -save_data political \
#     -src_vocab_size 20000

# Train the classifier
# python3 cnn_train.py \
#     -gpus 0 \
#     -data political.train.pt \
#     -save_model political_model \
#     -batch_size 128

BESTMODEL="political_model.pt"

# Test the classifier accuracy
python3 cnn_translate.py -gpu 0 -model $BESTMODEL -src ../../datasets/political_data/democratic_only.test.en -tgt 'democratic' -label0 democratic -label1 republican

