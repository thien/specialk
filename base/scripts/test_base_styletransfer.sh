# used to test style transfer on political dataset with transformers.

cd ..

# setup rebased datasets.
REP_PARTY="republican"
DEM_PARTY="democratic"
DEM_SAVENAME="models/nmt_fren_"$DEM_PARTY
REP_SAVENAME="models/nmt_fren_"$REP_PARTY
PT=".pt"

if [ -e $DEM_SAVENAME$PT ]
then
    echo "Already created rebased dataset for political data."
else
    BASE="models/nmt_fren_lg_lower_30k_v.pt"

    TRAIN_SRC="../datasets/political_data/"$DEM_PARTY"_only.train.fr"
    TRAIN_TGT="../datasets/political_data/"$DEM_PARTY"_only.train.en"
    VALID_SRC="../datasets/political_data/"$DEM_PARTY"_only.dev.fr"
    VALID_TGT="../datasets/political_data/"$DEM_PARTY"_only.dev.en"

    python3 rebase.py -base $BASE -train_src $TRAIN_SRC -train_tgt $TRAIN_TGT  -valid_src $VALID_SRC -valid_tgt $VALID_TGT -save_name $DEM_SAVENAME 

    TRAIN_SRC="../datasets/political_data/"$REP_PARTY"_only.train.fr"
    TRAIN_TGT="../datasets/political_data/"$REP_PARTY"_only.train.en"
    VALID_SRC="../datasets/political_data/"$REP_PARTY"_only.dev.fr"
    VALID_TGT="../datasets/political_data/"$REP_PARTY"_only.dev.en"

    python3 rebase.py -base $BASE -train_src $TRAIN_SRC -train_tgt $TRAIN_TGT  -valid_src $VALID_SRC -valid_tgt $VALID_TGT -save_name $REP_SAVENAME 
fi

MODEL="transformer"
EP=15
MODELDIM=512
BATCHSIZE=64

REF_ENCODER="models/nmt_fren_lg_lower_30k_v_base/encoder_epoch_15_accu_41.363.chkpt"
REF_DECODER="models/nmt_fren_lg_lower_30k_v_base/decoder_epoch_15_accu_41.363.chkpt"

DEM_DATASET=$DEM_SAVENAME$PT
REP_DATASET=$REP_SAVENAME$PT
DEM_MODELDIR="nmt_fren_lg_democrat_base"
REP_MODELDIR="nmt_fren_lg_republican_base"
ENCODER="encoder.chkpt"
DECODER="decoder.chkpt"

# democrat no freeze
python3 train.py \
    -checkpoint_encoder $REF_ENCODER \
    -checkpoint_decoder $REF_DECODER \
    -directory_name $DEM_MODELDIR \
    -data $DEM_DATASET \
    -model $MODEL \
    -epoch $EP \
    -d_word_vec $MODELDIM \
    -d_model $MODELDIM \
    -save_mode "best" \
    -batch_size $BATCHSIZE \
    -save_model \
    -log \
    -new_directory \
    -cuda 

python3 core/telegram.py -m "Finished training democrat no freeze."

# democrat freeze
python3 train.py \
    -checkpoint_encoder $REF_ENCODER \
    -checkpoint_decoder $REF_DECODER \
    -directory_name $DEM_MODELDIR"_freeze" \
    -data $DEM_DATASET \
    -model $MODEL \
    -epoch $EP \
    -d_word_vec $MODELDIM \
    -d_model $MODELDIM \
    -save_mode "best" \
    -batch_size $BATCHSIZE \
    -save_model \
    -log \
    -freeze_encoder \
    -new_directory \
    -cuda

python3 core/telegram.py -m "Finished training democrat freeze."

# republican no freeze
python3 train.py \
    -checkpoint_encoder $REF_ENCODER \
    -checkpoint_decoder $REF_DECODER \
    -directory_name $REP_MODELDIR \
    -data $REP_DATASET \
    -model $MODEL \
    -epoch $EP \
    -d_word_vec $MODELDIM \
    -d_model $MODELDIM \
    -save_mode "best" \
    -batch_size $BATCHSIZE \
    -save_model \
    -log \
    -new_directory \
    -cuda 

python3 core/telegram.py -m "Finished training republican no freeze."

# republican freeze
python3 train.py \
    -checkpoint_encoder $REF_ENCODER \
    -checkpoint_decoder $REF_DECODER \
    -directory_name $REP_MODELDIR"_freeze" \
    -data $REP_DATASET \
    -model $MODEL \
    -epoch $EP \
    -d_word_vec $MODELDIM \
    -d_model $MODELDIM \
    -save_mode "best" \
    -batch_size $BATCHSIZE \
    -save_model \
    -log \
    -freeze_encoder \
    -new_directory \
    -cuda 

python3 core/telegram.py -m "Finished training republican freeze."

# now we need to translate those sequences.
DEM_TESTSET="../datasets/political_data/democratic_only.test.fr"
REP_TESTSET="../datasets/political_data/republican_only.test.fr"
OUTPUT="outputs.txt"
EVALTXT="eval.txt"

# democrat no freeze
python3 translate.py \
    -model $MODEL \
    -checkpoint_encoder "models/"$DEM_MODELDIR"/"$ENCODER \
    -checkpoint_decoder "models/"$DEM_MODELDIR"/"$DECODER \
    -vocab $DEM_DATASET \
    -src $DEM_TESTSET \
    -output "results/baseline/democrat_nofreeze.txt" \
    -cuda

python3 core/telegram.py -m "Finished translating democrat no freeze."

# democrat freeze
python3 translate.py \
    -model $MODEL \
    -checkpoint_encoder "models/"$DEM_MODELDIR"_freeze/"$ENCODER \
    -checkpoint_decoder "models/"$DEM_MODELDIR"_freeze/"$DECODER \
    -vocab $DEM_DATASET \
    -src $DEM_TESTSET \
    -output "results/baseline/democrat_freeze.txt" \
    -cuda

python3 core/telegram.py -m "Finished translating democrat freeze."

# republican no freeze
python3 translate.py \
    -model $MODEL \
    -checkpoint_encoder "models/"$REP_MODELDIR"/"$ENCODER \
    -checkpoint_decoder "models/"$REP_MODELDIR"/"$DECODER \
    -vocab $REP_DATASET \
    -src $REP_TESTSET \
    -output "results/baseline/republican_nofreeze.txt" \
    -cuda

python3 core/telegram.py -m "Finished translating republican no freeze."

# republican freeze
python3 translate.py \
    -model $MODEL \
    -checkpoint_encoder "models/"$REP_MODELDIR"_freeze/"$ENCODER \
    -checkpoint_decoder "models/"$REP_MODELDIR"_freeze/"$DECODER \
    -vocab $REP_DATASET \
    -src $REP_TESTSET \
    -output "results/baseline/republican_freeze.txt" \
    -cuda

python3 core/telegram.py -m "Finished translating republican freeze."

# baseline democrat
python3 translate.py \
    -model $MODEL \
    -checkpoint_encoder $REF_ENCODER \
    -checkpoint_decoder $REF_ENCODER \
    -vocab $DEM_DATASET \
    -src $DEM_TESTSET \
    -output "results/baseline/baseline_democrat.txt" \
    -cuda

python3 core/telegram.py -m "Finished translating democrat baseline."

# baseline republican
python3 translate.py \
    -model $MODEL \
    -checkpoint_encoder $REF_ENCODER \
    -checkpoint_decoder $REF_ENCODER \
    -vocab $DEM_DATASET \
    -src $DEM_TESTSET \
    -output "results/baseline/baseline_republican.txt" \
    -cuda

python3 core/telegram.py -m "Finished translating republican baseline."


DEM_TESTSET_EN="../datasets/political_data/democratic_only.test.en"
REP_TESTSET_EN="../datasets/political_data/republican_only.test.en"

RESDIR="results/baseline/"

nlg-eval --hypothesis=$RESDIR"democrat_nofreeze.txt" --references=$DEM_TESTSET_EN > $RESDIR"democrat_nofreeze.stat.txt"

python3 core/telegram.py -m "Finished computing stats for democrat_nofreeze"

nlg-eval --hypothesis=$RESDIR"democrat_freeze.txt" --references=$DEM_TESTSET_EN > $RESDIR"democrat_freeze.stat.txt"

python3 core/telegram.py -m "Finished computing stats for democrat_freeze"

nlg-eval --hypothesis=$RESDIR"republican_nofreeze.txt" --references=$REP_TESTSET_EN > $RESDIR"republican_nofreeze.stat.txt"

python3 core/telegram.py -m "Finished computing stats for republican_nofreeze"

nlg-eval --hypothesis=$RESDIR"republican_freeze.txt" --references=$REP_TESTSET_EN > $RESDIR"republican_freeze.stat.txt"

python3 core/telegram.py -m "Finished computing stats for republican_freeze"

nlg-eval --hypothesis=$RESDIR"baseline_democrat.txt" --references=$DEM_TESTSET_EN > $RESDIR"baseline_democrat.stat.txt"

python3 core/telegram.py -m "Finished computing stats for baseline_democrat"

nlg-eval --hypothesis=$RESDIR"baseline_republican.txt" --references=$REP_TESTSET_EN > $RESDIR"baseline_republican.stat.txt"

python3 core/telegram.py -m "Finished computing stats for baseline_republican"