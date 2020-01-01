cd ..
# make models directory if it doesn't exist.
mkdir -p models

cd models
if [ -e wmt14.en-fr.joined-dict.transformer.pt ]
then
    echo "transformer already downloaded."
else
    wget https://dl.fbaipublicfiles.com/fairseq/models/wmt14.en-fr.joined-dict.transformer.tar.bz2
    tar xvjf wmt14.en-fr.joined-dict.transformer.tar.bz2
fi
