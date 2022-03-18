CHECKPOINT=checkpoints/checkpoint_best.pt
DSTORE_DIR=/iesl/local/adrozdov/knnlm_data
MAX_TOK=3072 # batch size
#MAX_TOK=6144 # batch size

mkdir -p $DSTORE_DIR

python build_dstore.py \
    --dstore_mmap $DSTORE_DIR/dstore \
    --dstore_size 103225485 \
    --faiss_index $DSTORE_DIR/knn.index \
    --num_keys_to_add_at_a_time 500000 \
    --starting_point 0 --dstore_fp16


