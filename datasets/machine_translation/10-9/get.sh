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
    # extract english language
    gunzip ./giga-fren.release2.fixed.en.gz 
    gunzip ./giga-fren.release2.fixed.fr.gz
    # normalise punct.
    cdir=$(pwd)
    cd ../../
    perl normalise_punctuation.perl -a -no_escape en -q < $cdir/giga-fren.release2.fixed.en > $cdir/tmp.en  
    rm $cdir/giga-fren.release2.fixed.en
    mv $cdir/tmp.en $cdir/giga-fren.release2.fixed.en
    # normalise punct.
    perl normalise_punctuation.perl -a -no_escape fr -q < $cdir/giga-fren.release2.fixed.fr > $cdir/tmp.fr
    rm $cdir/giga-fren.release2.fixed.fr
    mv $cdir/tmp.fr $cdir/giga-fren.release2.fixed.fr
    # go back to dir
    cd $cdir
    # filter gigatext.
    python3 ./filter_giga.py
    # delete ogs
    rm ./giga-fren.release2.fixed.en
    rm ./giga-fren.release2.fixed.fr
fi
