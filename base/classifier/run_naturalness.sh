

train_data="/home/t/Data/Files/Github/msc_project_model/datasets/naturalness/train_data.txt"
val_data="/home/t/Data/Files/Github/msc_project_model/datasets/naturalness/val_data.txt"
corpus_name="naturalness_political"
model_name="naturalness_political"
# Preprocess data
## Note: First create a dev.txt file such that
## 1. Each instance is on a new line
## 2. The label and the sentence are separated by a space
# python3 preprocess.py \
#     -train_src $train_data \
#     -label0 fake \
#     -label1 real \
#     -valid_src  $val_data \
#     -save_data $corpus_name \
#     -src_vocab_size 30000

# # Train the classifier
# python3 cnn_train.py \
#     -gpus 0 \
#     -data $corpus_name".train.pt" \
#     -save_model $model_name \
#     -batch_size 128

BESTMODEL=$model_name".pt"

testdata="/home/t/Data/Files/Github/msc_project_model/datasets/naturalness/out2/democratic_republican_test.txt"

outputpath="/home/t/Data/Files/Github/msc_project_model/datasets/naturalness/out2/testresults.txt"

# # Test the classifier accuracy
python3 cnn_translate.py \
    -gpu 0 \
    -model $BESTMODEL \
    -src $testdata \
    -tgt 'fake' \
    -output $outputpath \
    -label0 fake \
    -label1 real

