# Translate the political corpora into French with a trained EN-FR model.
#
# CHECKPOINT must be a Lightning .ckpt produced by
# `python3 -m specialk.trainers.train` (see doc/generation.md).

BASEDIR="/home/t/project_model/base/"
CHECKPOINT=${CHECKPOINT:?set CHECKPOINT to a trained EN-FR checkpoint}

cd $BASEDIR

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
           
# translate all political data into french.
for p in democratic republican
   do for src in $POLDATA_DIR$p"_only"*.en
       do python3 -m specialk.translate \
           --checkpoint "$CHECKPOINT" \
           --src "$src" \
           --output "$src.ts"
   done
done

# convert all ".ts" files to ".fr"
# rename 's/.en.ts/.fr/' $POLDATA_DIR*.ts
#for file in $POLDATA_DIR*.ts; do mv $file ${file/en.ts/.fr}; done

# FREN_CORP=$BASEDIR"models/nmt_fren_goldmaster_bpe.pt"
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


# # translate newspaper dataset


# # rebase newspaper dataset
