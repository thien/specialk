cd ..
perl tokenizer.perl -a -no-escape en -q < test_tokeniser/test_text.md > test_tokeniser/test_text_new.md
perl detokenizer.perl -a -no-escape en -q < test_tokeniser/test_text_new.md > test_tokeniser/test_text_back.md
