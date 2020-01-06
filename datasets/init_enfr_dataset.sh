#!/bin/bash
mkdir -p machine_translation
mkdir -p newspapers


# initiate.sh
# deals with creating an english-french NMT corpus.
# running this program will compute the whole thing for you.

# OUTPUT_EN="machine_translation/corpus_enfr.en"
# OUTPUT_FR="machine_translation/corpus_enfr.fr"


OUTPUT_EN="machine_translation/corpus_enfr_final.en"
OUTPUT_FR="machine_translation/corpus_enfr_final.fr"

#
#   GLOBAL VOICES DATASET
#

# download global_voices dataset
mkdir -p machine_translation/global_voices
if [ -e machine_translation/global_voices/dataset.tgz ]
then
    echo "already downloaded global voices corpus."
else
    echo -n "downloading global_voices.." 
    wget -O machine_translation/global_voices/dataset.tgz -q http://casmacat.eu/corpus/global-voices-tar-balls/training.tgz
    echo " done."
    echo "finished downloading the dataset."
fi

# extract dataset.
if [ -e machine_translation/global_voices/training/globalvoices.fr-en.en ]
then
    echo "global voices dataset is already extracted."
else
    cd machine_translation/global_voices
    tar -xvf dataset.tgz
    cd ../..
    echo "extracted dataset."
fi

# need to worry about cleaning the dataset because the english variant has a bunch of rubbish on it.
if [ -e machine_translation/global_voices/globalvoices.en ]
then
    echo "global voices translation corpus is already extracted."
else
    cd machine_translation/global_voices
    python3 clean.py -folder_path training -source fr -target en
    cd ../..
    echo "extracted corpus."
fi


#
#   HANSARDS
#

# check if we own the hansards dataset already.
mkdir -p machine_translation/hansards
if [ -e machine_translation/hansards/hansard.36 ]
then 
    echo "hansards already extracted."
else
    echo -n "downloading hansard corpus.. (around 83mb)"
    cd machine_translation/hansards
    wget -O house_training.tar -q http://www.isi.edu/natural-language/download/hansard/hansard.36.r2001-1a.house.debates.training.tar
    wget -O house_testing.tar -q http://www.isi.edu/natural-language/download/hansard/hansard.36.r2001-1a.house.debates.testing.tar
    wget -O senate_training.tar -q http://www.isi.edu/natural-language/download/hansard/hansard.36.r2001-1a.senate.debates.training.tar
    wget -O senate_testing.tar -q http://www.isi.edu/natural-language/download/hansard/hansard.36.r2001-1a.senate.debates.testing.tar
    echo " done."
    echo -n "extracting tarfiles.."
    tar xvf house_training.tar 
    tar xvf house_testing.tar
    tar xvf senate_training.tar
    tar xvf senate_testing.tar
    gzip -d -r hansard.36
    echo " done."
    cd ../..
fi

# deal with extracting the whole corpus.
if [ -e machine_translation/hansards/hansards.en ]
then 
    echo "hansards already scraped."
else
    echo -n "scraping hansards.."
    cd machine_translation/hansards
    python3 scrape.py
    echo " done."
    cd ../..
fi


#
#   EUROPARL
#

# check if the corpus is downloaded
mkdir -p machine_translation/europarl
if [ -e machine_translation/europarl/fr-en.tgz ]
then
    echo "europarl corpus already downloaded."
else
    echo -n "downloading europarl... "
    cd machine_translation/europarl
    wget -q http://www.statmt.org/europarl/v7/fr-en.tgz
    tar -xvf fr-en.tgz
    mv europarl-v7.fr-en.en europarl.en
    mv europarl-v7.fr-en.fr europarl.fr
    cd ../..
    echo "done."
fi

if [ -e machine_translation/commentary/news-commentary-v11.fr-en.fr ]
then
    echo "news commentary corpus already created."
else
    echo -n "Downloading News commentary dataset."
    cd machine_translation/commentary
    ./express.sh
    cd ../..
    echo "done."
fi

#
# GIGATEXT
#
if [ -e machine_translation/10-9/gigafren.en ]
then
    echo "already created gigatexts."
else
    cd machine_translation/10-9
    ./get.sh
    cd ../..
    echo "done."
fi

#
# MERGING DATASETS
#

if [ -e $OUTPUT_EN ]
then
    echo "corpus already merged."
else
    echo -n "merging datasets.."
    # cd machine_translation

    python3 merge.py \
        -left \
        machine_translation/europarl/europarl.en \
        machine_translation/global_voices/globalvoices.en \
        machine_translation/hansards/hansards.en \
        machine_translation/commentary/news-commentary-v11.fr-en.en \
        machine_translation/10-9/gigafren.en \
        -right \
        machine_translation/europarl/europarl.fr \
        machine_translation/global_voices/globalvoices.fr \
        machine_translation/hansards/hansards.fr \
        machine_translation/commentary/news-commentary-v11.fr-en.fr \
        machine_translation/10-9/gigafren.fr \
        -left_out \
        $OUTPUT_EN \
        -right_out \
        $OUTPUT_FR
    # cat europarl/europarl.en global_voices/globalvoices.en hansards/hansards.en > corpus_enfr.en.raw
    # cat europarl/europarl.fr global_voices/globalvoices.fr hansards/hansards.fr > corpus_enfr.fr.raw
    # # removing blank lines
    # sed '/^[[:space:]]*$/d' corpus_enfr.en.raw > corpus_enfr.en
    # sed '/^[[:space:]]*$/d' corpus_enfr.fr.raw > corpus_enfr.fr
    # rm corpus_enfr.en.raw
    # rm corpus_enfr.fr.raw
    echo " done."

    echo "Normalising Punctuation.."
    perl normalise_punctuation.perl -a -no_escape en < $OUTPUT_EN > tmp.en
    rm $OUTPUT_EN
    mv tmp.en $OUTPUT_EN
    perl normalise_punctuation.perl -a -no_escape fr < $OUTPUT_FR > tmp.fr
    rm $OUTPUT_FR
    mv tmp.fr $OUTPUT_FR

    echo "Sanitising Dataset.."
    python3 sanitise.py \
    -source_a \
    $OUTPUT_EN \
    -source_b \
    $OUTPUT_FR \
    -a_label en \
    -b_label fr 

    echo -n "number of sequences: "
    wc -l < $OUTPUT_EN
fi 

#
# TOKENISING
#
if [ -e $OUTPUT_EN".atok" ]
then 
    echo "corpus already tokenised."
else
    echo -n "performing sed.. "
    for l in en fr; do for f in machine_translation/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
    echo "done. "
    echo -n "tokenising corpus.. "
    for l in en fr
        do for f in machine_translation/*.$l
            do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok
        done
    done
    echo "done. "
fi


if [ -e $OUTPUT_EN".train.en.atok" ]
then
    echo "dataset already created."
else
    echo "splitting dataset into training, validation and test data."
    python3 splitter.py \
    -source_a $OUTPUT_EN".atok" \
    -source_b $OUTPUT_FR".atok" \
    -a_label en \
    -b_label fr \
    -verbose $true \
    -ratio 97:0.1:2.9
    echo "finished splitting dataset into training, validation, and test data."
fi
