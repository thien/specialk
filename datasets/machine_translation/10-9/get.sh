wget http://www.statmt.org/wmt10/training-giga-fren.tar
tar -xvf training-giga-fren.tar
gunzip giga-fren.release2.fixed.en.gz 
gunzip giga-fren.release2.fixed.fr.gz 
rm giga-fren.release2.fixed.fr.gz 
rm giga-fren.release2.fixed.en.gz
python3 filter_giga.py
rm giga-fren.release2.fixed.en
rm giga-fren.release2.fixed.fr