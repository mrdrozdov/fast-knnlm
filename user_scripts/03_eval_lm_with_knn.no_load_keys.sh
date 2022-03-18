CHECKPOINT=checkpoints/checkpoint_best.pt
DSTORE_DIR=/iesl/local/adrozdov/knnlm_data

python eval_lm.py data-bin/wikitext-103 \
    --path $CHECKPOINT \
    --sample-break-mode complete --max-tokens 3072 \
    --context-window 2560 --softmax-batch 64 \
    --gen-subset valid --dstore-filename $DSTORE_DIR/dstore \
    --indexfile $DSTORE_DIR/knn.index  \
    --model-overrides "{'knn_keytype': 'last_ffn_input'}" \
    --k 1024 --lmbda 0.25 --dstore-size 103225485 --knn-keytype last_ffn_input \
    --probe 32 --knnlm --fp16 --dstore-fp16 \
    --no-load-keys --knn-sim-func "do_not_recomp_l2"

