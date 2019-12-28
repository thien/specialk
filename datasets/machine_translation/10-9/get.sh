# download the tar and extract it.
if [ -e ./training-giga-fren.tar ]
then
    echo "already downloaded gigatext."
else
    wget http://www.statmt.org/wmt10/training-giga-fren.tar
fi

if [ -e ./gigafren.en ]
then 
    echo "already extracted gigatext."
else
    tar -xvf ./training-giga-fren.tar
    # extract each language
    gunzip ./giga-fren.release2.fixed.en.gz 
    gunzip ./giga-fren.release2.fixed.fr.gz 
    # filter gigatext.
    python3 ./filter_giga.py
    # delete ogs
    rm ./giga-fren.release2.fixed.en
    rm ./giga-fren.release2.fixed.fr
fi
