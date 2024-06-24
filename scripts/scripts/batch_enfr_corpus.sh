cd ..

if [ -e ../datasets/machine_translation/corpus_enfr.test.en ]
then
    FILEPATH="../datasets/machine_translation/"
    TRAIN_EN=$FILEPATH"corpus_enfr.train.en"
    TRAIN_FR=$FILEPATH"corpus_enfr.train.fr"
    VALID_EN=$FILEPATH"corpus_enfr.val.en"
    VALID_FR=$FILEPATH"corpus_enfr.val.fr"
    FORMAT="word"
    MAXLEN="70"

    PTF=".pt"
    B="_base"
    
    SM="50000"
    MD="500000"
    LG="1500000"

    m="models/"

    VOCAB_SIZE="30000"

    EXT="_lower_30k_v"

    # VSIZE=""
    SM_ENFR="nmt_enfr_50k"$EXT
    MD_ENFR="nmt_enfr_500k"$EXT
    LG_ENFR="nmt_enfr_lg"$EXT
    FR_ENFR="nmt_enfr_full"$EXT

    SM_FREN="nmt_fren_50k"$EXT
    MD_FREN="nmt_fren_500k"$EXT
    LG_FREN="nmt_fren_lg"$EXT
    FR_FREN="nmt_fren_full"$EXT

    SAVENAME_SM_ENFR=$m$SM_ENFR
    SAVENAME_MD_ENFR=$m$MD_ENFR
    SAVENAME_LG_ENFR=$m$LG_ENFR
    SAVENAME_FR_ENFR=$m$FR_ENFR

    SAVENAME_SM_FREN=$m$SM_FREN
    SAVENAME_MD_FREN=$m$MD_FREN
    SAVENAME_LG_FREN=$m$LG_FREN
    SAVENAME_FR_FREN=$m$FR_FREN

    if [ -e $SAVENAME_SM_FREN$PTF ]
    then
        echo "small corpus already created."
    else
        # setup small corpus
        python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_FR -valid_src $VALID_EN -valid_tgt $VALID_FR -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_SM_ENFR -max_train_seq $SM -vocab_size $VOCAB_SIZE

        python3 preprocess.py -train_src $TRAIN_FR -train_tgt $TRAIN_EN -valid_src $VALID_FR -valid_tgt $VALID_EN -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_SM_FREN -max_train_seq $SM -vocab_size $VOCAB_SIZE
    fi


    if [ -e $SAVENAME_MD_FREN$PTF ]
    then
        echo "medium corpus already created."
    else
        # setup medium corpus
        python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_FR -valid_src $VALID_EN -valid_tgt $VALID_FR -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_MD_ENFR -max_train_seq $MD -vocab_size $VOCAB_SIZE

        python3 preprocess.py -train_src $TRAIN_FR -train_tgt $TRAIN_EN -valid_src $VALID_FR -valid_tgt $VALID_EN -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_MD_FREN -max_train_seq $MD -vocab_size $VOCAB_SIZE
    fi


    if [ -e $SAVENAME_LG_FREN$PTF ]
    then
        echo "large corpus already created."
    else
        # setup large corpus
        python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_FR -valid_src $VALID_EN -valid_tgt $VALID_FR -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_LG_ENFR -max_train_seq $LG -vocab_size $VOCAB_SIZE

        python3 preprocess.py -train_src $TRAIN_FR -train_tgt $TRAIN_EN -valid_src $VALID_FR -valid_tgt $VALID_EN -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_LG_FREN -max_train_seq $LG -vocab_size $VOCAB_SIZE
    fi


    if [ -e $SAVENAME_FR_FREN$PTF ]
    then
        echo "xl corpus already created."
    else
        python3 preprocess.py -train_src $TRAIN_EN -train_tgt $TRAIN_FR -valid_src $VALID_EN -valid_tgt $VALID_FR -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_FR_ENFR  -vocab_size $VOCAB_SIZE

        python3 preprocess.py -train_src $TRAIN_FR -train_tgt $TRAIN_EN -valid_src $VALID_FR -valid_tgt $VALID_EN -format $FORMAT -max_len $MAXLEN -save_name $SAVENAME_FR_FREN  -vocab_size $VOCAB_SIZE
    fi

    # # train models

    
    MODEL="transformer"
    EP=15
    MODELDIM=512
    BSIZE=64

    # SMALL MODELS

    # python3 train.py \
    #     -log $true \
    #     -batch_size $BSIZE \
    #     -model $MODEL \
    #     -epoch $EP \
    #     -d_word_vec $MODELDIM \
    #     -d_model $MODELDIM \
    #     -cuda \
    #     -data $SAVENAME_SM_ENFR$PTF \
    #     -directory_name $SM_ENFR$B \
    #     -save_model \
    #     -save_mode best 

    # python3 core/telegram.py -m "Finished training en-fr small models."
    
    # python3 train.py \
    #     -log $true \
    #     -batch_size $BSIZE \
    #     -model $MODEL \
    #     -epoch $EP \
    #     -d_word_vec $MODELDIM \
    #     -d_model $MODELDIM \
    #     -cuda \
    #     -data $SAVENAME_SM_FREN$PTF \
    #     -directory_name $SM_FREN$B \
    #     -save_model \
    #     -save_mode best 

    # python3 core/telegram.py -m "Finished training fr-en small models."

    # medium

    # MEDIUM

    python3 train.py \
        -log $true \
        -batch_size $BSIZE \
        -model $MODEL \
        -epoch $EP \
        -d_word_vec $MODELDIM \
        -d_model $MODELDIM \
        -cuda \
        -data $SAVENAME_MD_ENFR$PTF \
        -directory_name $MD_ENFR$B \
        -save_model \
        -save_mode all

    python3 core/telegram.py -m "Finished training en-fr medium models."
    
    python3 train.py \
        -log $true \
        -batch_size $BSIZE \
        -model $MODEL \
        -epoch $EP \
        -d_word_vec $MODELDIM \
        -d_model $MODELDIM \
        -cuda \
        -data $SAVENAME_MD_FREN$PTF \
        -directory_name $MD_FREN$B \
        -save_model \
        -save_mode all

    python3 core/telegram.py -m "Finished training fr-en medium models."

    # large

    python3 train.py \
        -log $true \
        -batch_size $BSIZE \
        -model $MODEL \
        -epoch $EP \
        -d_word_vec $MODELDIM \
        -d_model $MODELDIM \
        -cuda \
        -data $SAVENAME_LG_ENFR$PTF \
        -directory_name $LG_ENFR$B \
        -save_model \
        -save_mode all

    python3 core/telegram.py -m "Finished training en-fr large models."
    
    python3 train.py \
        -log $true \
        -batch_size $BSIZE \
        -model $MODEL \
        -epoch $EP \
        -d_word_vec $MODELDIM \
        -d_model $MODELDIM \
        -cuda \
        -data $SAVENAME_LG_FREN$PTF \
        -directory_name $LG_FREN$B \
        -save_model \
        -save_mode all

    python3 core/telegram.py -m "Finished training fr-en large models."

    # python3 train.py -log $true -batch_size $BSIZE -save_model -model $MODEL -epoch $EP -d_word_vec $MODELDIM -d_model $MODELDIM -cuda -data $SAVENAME_FR_ENFR$PTF -directory_name $FR_ENFR$B

    # python3 core/telegram.py -m "Finished transferring XL en-fr"

    # python3 train.py -log $true -batch_size $BSIZE -save_model -model $MODEL -epoch $EP -d_word_vec $MODELDIM -d_model $MODELDIM -cuda -data $SAVENAME_LG_ENFR$PTF -directory_name $LG_ENFR$B

    # python3 core/telegram.py -m "Finished transferring L en-fr"

    # python3 train.py -log $true -batch_size $BSIZE -save_model -model $MODEL -epoch $EP -d_word_vec $MODELDIM -d_model $MODELDIM -cuda -data $SAVENAME_MD_ENFR$PTF -directory_name $MD_ENFR$B

    # python3 core/telegram.py -m "Finished transferring M en-fr"

    # python3 train.py -log $true -batch_size $BSIZE -save_model -model $MODEL -epoch $EP -d_word_vec $MODELDIM -d_model $MODELDIM -cuda -data $SAVENAME_SM_ENFR$PTF -directory_name $SM_ENFR$B

    # python3 core/telegram.py -m "Finished transferring S en-fr"

    # # FR-EN

    # python3 train.py -log $true -batch_size $BSIZE -save_model -model $MODEL -epoch $EP -d_word_vec $MODELDIM -d_model $MODELDIM -cuda -data $SAVENAME_SM_FREN$PTF -directory_name $SM_FREN$B

    # python3 core/telegram.py -m "Finished transferring S fr-en"


else
    echo "You need to create the corpus."
fi
