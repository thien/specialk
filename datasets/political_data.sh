#!/bin/bash
mkdir -p political_data

# download dataset
if [ -e political_data/political_data.tar ]
then
    echo "Political data already downloaded."
else
    echo -n "Downloading political dataset.."
    wget -O political_data/political_data.tar -q http://tts.speech.cs.cmu.edu/style_models/political_data.tar
    echo " done."
fi
# extract dataset.
if [ -e political_data/political_data/ ]
then 
    echo "Political data tar is already extracted."
else
    cd political_data
    tar -xvf political_data.tar
    echo "tar data extracted."
fi
# note that the dataset is already tokenised.


# note that you can't make the preprocessed dataset until you have translated them.