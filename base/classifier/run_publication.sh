
# Preprocess data
## Note: First create a dev.txt file such that
## 1. Each instance is on a new line
## 2. The label and the sentence are separated by a space
LAB0="popular"
LAB1="quality"

TRAIN_DATA="../../datasets/newspapers/classifier.train.en"
VALID_DATA="../../datasets/newspapers/classifier.val.en"
TEST_DATA="../../datasets/political_data/democratic_only.test.en"

CORPUSNAME="publication"
MODELNAME="publication_model"

MAX_SEQ_LEN=200
VOCABSIZE=40000

# python3 preprocess.py \
#     -train_src $TRAIN_DATA \
#     -valid_src $VALID_DATA \
#     -label0 $LAB0 \
#     -label1 $LAB1 \
#     -save_data $CORPUSNAME \
#     -src_vocab_size $VOCABSIZE \
#     -seq_length $MAX_SEQ_LEN \
#     -lower

MODELNAME="publication_model_test"

BESTMODEL="publication_model_acc_91.76_loss_0.00_e10.pt"
# Train the classifier
python3 cnn_train.py \
    -gpus 0 \
    -data $CORPUSNAME.train.pt \
    -save_model $MODELNAME \
    -sequence_length $MAX_SEQ_LEN \
    -batch_size 64 \
    -epochs 1 \
&& python3 cnn_translate.py\
    -gpu 0 \
    -model $BESTMODEL \
    -src $TEST_DATA \
    -tgt $LAB0 \
    -label0 $LAB0 \
    -label1 $LAB1

