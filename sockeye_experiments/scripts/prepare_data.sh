set -e

raw_data=${1:-"/srv/scratch2/kew/fairseq_materials/rrgen_transformers/de/SG0.7_SL3_LR1.8_UP3/raw/"}

timestamp=$(date +'%y-%m-%d')
N=20
out_dir=$raw_data/../SE_$timestamp/
head_example_dir=$out_dir/head$N/

src="review.sp"
tgt="response.sp"
src_feat="sentiment_seq.sp"


if [ -d "$head_example_dir" ]; then
    echo "removing existing output directory..."
    rm -r $head_example_dir
fi

mkdir -p "$head_example_dir"
# # cut 20 training examples
head -n 20 "$raw_data/train.$src" > "$head_example_dir/train.$src" # src
head -n 20 "$raw_data/train.$src_feat" > "$head_example_dir/train.$src_feat" # src feats
head -n 20 "$raw_data/train.$tgt" > "$head_example_dir/train.$tgt" # tgt

head -n 20 "$raw_data/valid.$src" > "$head_example_dir/valid.$src" # src
head -n 20 "$raw_data/valid.$src_feat" > "$head_example_dir/valid.$src_feat" # src feats
head -n 20 "$raw_data/valid.$tgt" > "$head_example_dir/valid.$tgt" # tgt

head -n 20 "$raw_data/test.$src" > "$head_example_dir/test.$src" # src
head -n 20 "$raw_data/test.$src_feat" > "$head_example_dir/test.$src_feat" # src feats
head -n 20 "$raw_data/test.$tgt" > "$head_example_dir/test.$tgt" # tgt

# create dictionaries from full dataset
python -m sockeye.prepare_data \
    --source "$raw_data/train.$src" \
    --target "$raw_data/train.$tgt" \
    --source-factors "$raw_data/train.$src_feat" \
    --output "$head_example_dir/prepared_data" \
    --shared-vocab --max-seq-len 511