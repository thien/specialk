cd ..
perl tokenizer.perl -a -no-escape en -q < test_perl/test_text.md > test_perl/test_text_new.md
perl detokenizer.perl -a -no-escape en -q < test_perl/test_text_new.md > test_perl/test_text_back.md
perl normalise_punctuation.perl -l fr < test_perl/test_text.md > test_perl/test_text_p.md
