# assumes you transferred the GM models from ./snorlax_to_casper.sh

BASEDIR="/home/t/project_model/base/"
MODELDIR=$BASEDIR"models/"

cd $BASEDIR

ENFR_DIRNAME="enfr_bpe_gold_master"
FREN_DIRNAME="fren_bpe_gold_master"
VOCAB=$MODELDIR"nmt_enfr_goldmaster_bpe_rebase_for_mem.pt"
# translate political dataset
ENFR_BASEDIR="models/"$ENFR_DIRNAME"/"
ENCODER='encoder_epoch_1.chkpt'
DECODER='decoder_epoch_1.chkpt'
OUTPUT="outputs.txt"

# en -> fr
POLDATA_DIR="../datasets/political_data/"

# preprocess
# for p in democratic republican
#     do for c in dev train 
#         do for src in $POLDATA_DIR$p"_only."$c.en
#             do sed -i "s/^[A-Za-z0-9_]*\ //" $src
#         done
#     done
#     # cp $POLDATA_DIR$p"_only.test".en $POLDATA_DIR$p"_only.test".en.b
# done 
           
# translate all political data info french.
for p in democratic republican
   do for src in $POLDATA_DIR$p"_only"*.en
       do python3 translate.py \
           -model transformer \
           -checkpoint_encoder $ENFR_BASEDIR$ENCODER \
           -checkpoint_decoder $ENFR_BASEDIR$DECODER \
           -vocab $VOCAB \
           -src $src \
           -output $src".ts" \
           -cuda; python3 core/telegram.py -m "finished translating $src"
   done
done

# convert all ".ts" files to ".fr"
# rename 's/.en.ts/.fr/' $POLDATA_DIR*.ts
#for file in $POLDATA_DIR*.ts; do mv $file ${file/en.ts/.fr}; done

# FREN_CORP=$MODELDIR"nmt_fren_goldmaster_bpe.pt"
# # rebase political dataset
# for p in democratic
#     do python3 rebase.py \
#         -base $FREN_CORP \
#         -train_src $POLDATA_DIR$p"_only.train.fr" \
#         -train_tgt $POLDATA_DIR$p"_only.train.en.b" \
#         -valid_src $POLDATA_DIR$p"_only.dev.fr" \
#         -valid_tgt $POLDATA_DIR$p"_only.dev.en.b" \
#         -save_name "models/nmt_fren_goldmaster_bpe_"$p 
# done
# python3 core/telegram.py -m "finished rebasing political datasets."


# # translate newspaper dataset


# # rebase newspaper dataset

