
##########
# The first example shows how to build the democratic generator.
# 1. We first translate the democratic data from English to French
# 2. We then train the democratic style generator
#########

# Translate the democratic data from English to French
# Note: Use onmt.Translator when using the English-French translation system

ENGFR="../models/translation/english_french/english_french.pt"
FRENG="../models/translation/french_english/french_english.pt"
DEMTRAINEN="../data/political_data/democratic_only.train.en"
DEMTRAINFR="../data/political_data/democratic_only.train.fr"


# python preprocess.py -train_src TRAIN_SOURCE_FILE -train_tgt TRAIN_TARGET_FILE -valid_src VALID_SOURCE_FILE -valid_tgt VALID_TARGET_FILE -save_data DATA_NAME

# Translate the democratic data from English to French
# Note: Use onmt.Translator when using the English-French translation system

# Alternatively, you can just use the datasets provided when you downloaded the corpus.

# python3 translate.py -gpu 0 -model $ENGFR -src ../data/political_data/democratic_only.train.en -output ../data/political_data/democratic_only.train.fr -replace_unk $true

# python3 translate.py -gpu 0 -model $ENGFR -src ../data/political_data/democratic_only.dev.en -output ../data/political_data/democratic_only.dev.fr -replace_unk $true

# python3 translate.py -gpu 0 -model $ENGFR -src ../data/political_data/democratic_only.test.en -output ../data/political_data/democratic_only.test.fr -replace_unk $true

# python3 translate.py -gpu 0 -model $FRENG -src ../data/political_data/democratic_only.test.fr -output  ../data/political_data/democratic_only.test_en.new -replace_unk $true

# python3 translate.py -gpu 0 -model $ENGFR -src ../data/political_data/republican_only.dev.en -output ../data/political_data/republican_only.dev.fr -replace_unk $true

# python3 translate.py -gpu 0 -model $ENGFR -src ../data/political_data/republican_only.train.en -output ../data/political_data/republican_only.train.fr -replace_unk $true

# python3 translate.py -gpu 0 -model $ENGFR -src ../data/political_data/republican_only.test.en -output ../data/political_data/republican_only.test.fr -replace_unk $true
# #------------------------

# # Preprocess democratic french source and democratic english target data for the generator
# python3 preprocess.py -train_src ../data/political_data/democratic_only.train.fr -train_tgt ../data/political_data/democratic_only.train.en -valid_src ../data/political_data/democratic_only.dev.fr -valid_tgt ../data/political_data/democratic_only.dev.en -save_data data/democratic_generator -src_vocab ../models/translation/french_english/french_english_vocab.src.dict -tgt_vocab ../models/classifier/political_classifier/political_classifier_vocab.src.dict -seq_len 50

# # note: this saves democratic_generator.train.pt to data/ so you'll need to move it.

# # # Train the democratic style generator
# python3 train_decoder.py -gpus 0 -data data/democratic_generator.train.pt -save_model trained_models/democratic_generator -classifier_model ../models/classifier/political_classifier/political_classifier.pt -encoder_model ../models/translation/french_english/french_english.pt -tgt_label 0

# # # Translate the republican test set using the best democratic generator
python3 translate.py -gpu 0 -encoder_model $FRENG -decoder_model ../models/style_generators/democratic_generator.pt -src ../data/political_data/republican_only.test.fr -output trained_models/republican_democratic_test.txt -replace_unk $true

# python3 translate.py -gpu 0 -encoder_model $FRENG -decoder_model ../models/style_generators/democratic_generator.pt -src ../data/political_data/republican_only.dev.fr -output trained_models/republican_democratic_dev.txt -replace_unk $true

# python3 translate.py -gpu 0 -encoder_model $FRENG -decoder_model ../models/style_generators/democratic_generator.pt -src ../data/political_data/republican_only.train.fr -output trained_models/republican_democratic_train.txt -replace_unk $true

# python3 translate.py -gpu 0 -encoder_model $FRENG -decoder_model ../models/style_generators/republican_generator.pt -src ../data/political_data/democratic_only.train.fr -output trained_models/democratic_republican_train.txt -replace_unk $true

# python3 translate.py -gpu 0 -encoder_model $FRENG -decoder_model ../models/style_generators/republican_generator.pt -src ../data/political_data/democratic_only.dev.fr -output trained_models/democratic_republican_dev.txt -replace_unk $true

# python3 translate.py -gpu 0 -encoder_model $FRENG -decoder_model ../models/style_generators/republican_generator.pt -src ../data/political_data/democratic_only.test.fr -output trained_models/democratic_republican_test.txt -replace_unk $true

# python3 translate.py -gpu 0 -model $FRENG -src ../data/political_data/republican_only.test.fr -output trained_models/republican_democratic.txt -replace_unk $true -verbose $true
