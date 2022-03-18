CHECKPOINT=checkpoints/checkpoint_best.pt
DSTORE_DIR=/iesl/local/adrozdov/knnlm_data.test
MAX_TOK=3072 # batch size
#MAX_TOK=6144 # batch size

mkdir -p $DSTORE_DIR

python eval_lm.py data-bin/wikitext-103 \
    --path $CHECKPOINT \
    --sample-break-mode complete --max-tokens $MAX_TOK \
    --softmax-batch 1024 --gen-subset test \
    --context-window 2560 --no-min-context  \
    --dstore-mmap $DSTORE_DIR/dstore --knn-keytype 'last_ffn_input' \
    --dstore-size 245569 --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --save-knnlm-dstore --fp16 --dstore-fp16


