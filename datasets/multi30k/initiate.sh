cd ..

# echo -n "performing sed.. "
# for l in en de; do for f in multi30k/*.$l; do if [[ "$f" != *"test"* ]]; then sed -i "$ d" $f; fi;  done; done
# echo "done. "
echo -n "tokenising corpus.. "
for l in en de
    do for f in multi30k/*.$l
        do perl tokenizer.perl -a -no-escape -l $l -q  < $f > $f.atok
    done
done
echo "done. "