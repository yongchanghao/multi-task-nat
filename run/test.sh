# -------- manual configuration ---------
# see train.py for specific meaning
NAME=#
WRITE=#
DATA_BIN=#
SRC=#
TGT=#

# -------- normally no need to change ---------
CHECKPOINT=$WRITE/$NAME/ckpts
DECODING=$WRITE/$NAME/decoding/test
mkdir $DECODING
# Note that you have to obtain the averaged modle
# following the instruction in the README file.
ckpt=averaged.pt 

# -------- specific arguments ---------
CUDA_VISIBLE_DEVICES=0 python generate.py \
    $DATA_BIN \
    --fp16 \
    -s $SRC -t $TGT \
    --gen-subset test \
    --max-tokens 2048 \
    --task translation_mt \
    --generator nat \
    --path $CHECKPOINT/$ckpt \
    --iter-decode-max-iter 10 \
    --iter-decode-force-max-iter \
    --remove-bpe \
    --iter-decode-with-beam 5 \
    --valid-decoding-path $DECODING \
    --decoding-path $DECODING \
    --multi-bleu-path ./scripts \
    --num-ref $DATA=1 \
    |& tee $DECODING/$ckpt.gen
