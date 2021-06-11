#! /bin/bash

raw_data=${1:-"/srv/scratch2/kew/fairseq_materials/rrgen_transformers/de/SG0.7_SL3_LR1.8_UP3/raw/"}

timestamp=$(date +'%y-%m-%d')
N=20
out_dir=$raw_data/../SE_$timestamp/
head_example_dir=$out_dir/head$N/


scripts=`dirname "$0"`
base=$scripts/..

mkdir -p $base/models

num_threads=1
model_name=baseline

python -m sockeye.train \
    --source "$head_example/train.$src" --source-vocab "$head_example/prepared_data/vocab.src.0.json" \
    --source-factors "$head_example/train.$src_feat" --source-factor-vocabs "$head_example/prepared_data/vocab.src.1.json" \
    --target "$head_example/train.$tgt" --target-vocab "$head_example/prepared_data/vocab.trg.0.json" \
    --output "$head_example/model_dir/" \
    --validation-source "$head_example/valid.$src" \
    --validation-target "$head_example/valid.$tgt" \
    --validation-source-factors "$head_example/valid.$src_feat" \
    --source-factors-num-embed 128 \
    --shared-vocab \
    --initial-learning-rate 1e-4 --optimizer adam \
    --max-num-checkpoint-not-improved 3 \
    --batch-size 4 \
    --batch-type sentence \
    --device-ids $GPU



# ##################################

# OMP_NUM_THREADS=$num_threads python -m sockeye.train \
#       -o $base/models/$model_name  \
# 			-s $base/data/train.bpe.de \
# 			-t $base/data/train.bpe.en \
# 			-vs $base/data/dev.bpe.de \
#       -vt $base/data/dev.bpe.en \
#       --seed=1 \
#       --batch-type=word \
#       --batch-size=4096 \
#       --checkpoint-frequency=4000 \
#       --device-ids=0 \
#       --decode-and-evaluate-device-id 0 \
#       --embed-dropout=0:0 \
#       --encoder=transformer \
#       --decoder=transformer \
#       --num-layers=6:6 \
#       --transformer-model-size=512 \
#       --transformer-attention-heads=8 \
#       --transformer-feed-forward-num-hidden=2048 \
#       --transformer-preprocess=n \
#       --transformer-postprocess=dr \
#       --transformer-dropout-attention=0.1 \
#       --transformer-dropout-relu=0.1 \
#       --transformer-dropout-prepost=0.1 \
#       --transformer-positional-embedding-type fixed \
#       --fill-up=replicate \
#       --max-seq-len=100:100 \
#       --label-smoothing 0.1 \
#       --weight-tying \
#       --weight-tying-type=src_trg_softmax \
#       --num-embed 512:512 \
#       --num-words 50000:50000 \
#       --word-min-count 1:1 \
#       --optimizer=adam \
#       --optimized-metric=perplexity \
#       --clip-gradient=-1 \
#       --initial-learning-rate=0.0001 \
#       --learning-rate-reduce-num-not-improved=8 \
#       --learning-rate-reduce-factor=0.7 \
#       --learning-rate-scheduler-type=plateau-reduce \
#       --learning-rate-warmup=0 \
#       --max-num-checkpoint-not-improved=32 \
#       --min-num-epochs=0 \
#       --max-updates 1001000 \
#       --weight-init xavier \
#       --weight-init-scale 3.0 \
#       --weight-init-xavier-factor-type avg
