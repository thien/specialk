#!/bin/bash
mkdir -p machine_translation
mkdir -p newspapers


#
#   GLOBAL VOICES DATASET
#

# download global_voices dataset
if [ -e machine_translation/global_voices/dataset.tgz ]
then
    echo "already downloaded global voices corpus."
else
    echo -n "downloading global_voices.." 
    wget -O machine_translation/global_voices/dataset.tgz -q http://casmacat.eu/corpus/global-voices-tar-balls/training.tgz
    echo " done."
    # need to extract the corpus.
    echo "finished extracting the dataset."
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