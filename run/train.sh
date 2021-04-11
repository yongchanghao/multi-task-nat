export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# -------- manual configuration ---------
NAME=# {{whatever, a fancy name for a single experiment, e.g. mtnat, bert-xxxxxl, GPT-5, ...}}
WRITE=# {{root directory for all experiments, e.g. $HOME/my_experiments_dir }}
DATA=# {{your original text data, for evaluating bleu during training}}
DATA_BIN=# {{preprocessed data here}}
SRC=# {{suffix of the source language}}
TGT=# {{suffix of the target language}}


# -------- normally no need to change ---------
CHECKPOINT="$WRITE/$NAME/ckpts"
TB="$WRITE/$NAME/tensorboard"
VALID_PATH="$WRITE/$NAME/decoding"
mkdir $VALID_PATH
sed -r 's/(@@ )|(@@ ?$)//g' < $DATA/valid.$SRC > $VALID_PATH/valid.$SRC
sed -r 's/(@@ )|(@@ ?$)//g' < $DATA/valid.$TGT > $VALID_PATH/valid.$TGT


# -------- specific arguments ---------
python train.py \
    $DATA_BIN \
    --fp16 \
    -s $SRC -t $TGT \
    --save-dir $CHECKPOINT \
    --ddp-backend=no_c10d \
    --task translation_mt \
    --criterion mt_loss \
    --arch mt_transformer \
    --noise random_mask \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.1 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --share-all-embeddings \
    --at-weight 0.5 \
    --nat-weight 0.5 \
    --at-drop-rate 0 \
    --nat-drop-rate 0 \
    --log-format 'simple' --log-interval 100 \
    --max-tokens 16000 \
    --update-freq 1 \
    --save-interval-updates 2000 \
    --apply-bert-init \
    --max-update 300000 \
    --no-epoch-checkpoints \
    --quiet \
    --max-sentences-valid 128 \
    --all-gather-list-size 522240 \
    --num-ref $DATA=1 \
    --valid-decoding-path $VALID_PATH\
    --share-encoder \
    --remove-bpe \
    --multi-bleu-path ./scripts \
    --selection-criterion nat \
    --tensorboard-logdir $TB