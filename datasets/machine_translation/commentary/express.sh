wget -q http://www.casmacat.eu/corpus/news-commentary/news-commentary-v11.fr-en.xliff.gz
gunzip news-commentary-v11.fr-en.xliff.gz 
python3 convert.py
rm *.xliff
