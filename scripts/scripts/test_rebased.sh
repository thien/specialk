cd ..


BASE="models/nmt_fren_lg_lower_30k_v.pt"

PARTY="democratic"
TRAIN_SRC="../datasets/political_data/"$PARTY"_only.train.fr"
TRAIN_TGT="../datasets/political_data/"$PARTY"_only.train.en"
VALID_SRC="../datasets/political_data/"$PARTY"_only.dev.fr"
VALID_TGT="../datasets/political_data/"$PARTY"_only.dev.en"
SAVENAME="models/nmt_fren_"$PARTY

python3 rebase.py -base $BASE -train_src $TRAIN_SRC -train_tgt $TRAIN_TGT  -valid_src $VALID_SRC -valid_tgt $VALID_TGT -save_name $SAVENAME 

PARTY="republican"
TRAIN_SRC="../datasets/political_data/"$PARTY"_only.train.fr"
TRAIN_TGT="../datasets/political_data/"$PARTY"_only.train.en"
VALID_SRC="../datasets/political_data/"$PARTY"_only.dev.fr"
VALID_TGT="../datasets/political_data/"$PARTY"_only.dev.en"
SAVENAME="models/nmt_fren_"$PARTY

python3 rebase.py -base $BASE -train_src $TRAIN_SRC -train_tgt $TRAIN_TGT  -valid_src $VALID_SRC -valid_tgt $VALID_TGT -save_name $SAVENAME 
