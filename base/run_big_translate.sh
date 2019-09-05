MODEL_DIR=/home/t/Downloads/wmt14.en-fr.joined-dict.transformer

clear

fairseq-interactive \
    --path $MODEL_DIR/model.pt $MODEL_DIR \
    --beam 5 --source-lang en --target-lang fr \
    --tokenizer moses \
    --bpe subword_nmt \
    --bpe-codes $MODEL_DIR/bpecodes \
    --batch-size 64 \
    --buffer-size 64 \
    --remove-bpe \
    --input /home/t/project_model/datasets/newspapers/quality_sent.en.atok \
    --quiet $true \
#     --results-path /home/t/Downloads/wmt14.en-fr.joined-dict.transformer


# TEXT=/home/t/project_model/datasets/newspapers/quality_sent.en.atok 
# fairseq-preprocess \
#     --source-lang en --target-lang fr \
#     --testpref $TEXT \
#     --destdir /home/t/qual_np --thresholdtgt 0 --thresholdsrc 0 \
#     --workers 20

# fairseq-generate /home/t/qual_np \
#     --path $MODEL_DIR/model.pt $MODEL_DIR \
#     --batch-size 128 --beam 5 --remove-bpe
