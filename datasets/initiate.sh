#!/bin/bash
echo "what"
mkdir -p machine_translation
mkdir -p newspapers

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
if [-e machine_translation/global_voices/training/globalvoices.fr-en.en]
then
    echo "dataset is already extracted."
else
    tar -xvf machine_translation/global_voices/dataset.tgz
    echo "extracted dataset."
fi

# need to worry about cleaning the dataset because the english variant has a bunch of rubbish on it.
if [-e machine_translation/global_voices/globalvoices.en]
then
    echo "translation corpus is already extracted."
else
    cd machine_translation/global_voices
    python3 clean.py -folder_path training -source fr -target en
    cd ../..
    echo "extracted corpus."
fi

