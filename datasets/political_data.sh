#!/bin/bash
# mkdir -p political_data

if [ -e political_data/classtrain.txt ]
then
    echo "Political data already extracted."
else
    if [ -e political_data.tar ]
    then
        # need to extract it.
        tar -xvf political_data.tar
        echo "tar data extracted."
    else
        # download the dataset
        echo -n "Downloading political dataset.."
        wget -O political_data.tar -q http://tts.speech.cs.cmu.edu/style_models/political_data.tar
        echo " done."
        tar -xvf political_data.tar
        echo "tar data extracted."
    fi
fi

if [ -e political_data.tar ]
then
    rm political_data.tar
fi

# # download dataset
# if [ -e political_data/political_data.tar ]
# then
#     echo "Political data already downloaded."
# else
#     echo -n "Downloading political dataset.."
#     wget -O political_data/political_data.tar -q http://tts.speech.cs.cmu.edu/style_models/political_data.tar
#     echo " done."
# fi
# # extract dataset.
# if [ -e political_data/political_data/ ]
# then 
#     echo "Political data tar is already extracted."
# else
#     cd political_data
#     tar -xvf political_data.tar
#     echo "tar data extracted."
# fi

# # note that the dataset is already tokenised.
