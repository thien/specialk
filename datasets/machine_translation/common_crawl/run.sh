wget http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz 
tar -xvf training-parallel-commoncrawl.tgz
mkdir temp
mv commoncrawl.* temp
cd temp
mv commoncrawl.fr-en.* ../
cd ..
rm -rf temp
rm *.annotation
