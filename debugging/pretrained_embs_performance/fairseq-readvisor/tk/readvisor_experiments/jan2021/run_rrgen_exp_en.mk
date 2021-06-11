
# PREPROCESSING
MINI_PREP_DATA: /srv/scratch2/kew/fairseq_materials/rrgen_012021/en/mini
	fairseq-preprocess \
	--task rrgen_translation \
	--trainpref $</bpe/train \
	--validpref $</bpe/valid \
	--testpref $</bpe/test \
	--source-lang review \
	--target-lang response \
	--sent-ext alpha_sentiment \
	--rate-ext rating \
	--cate-ext domain \
	--len-ext review_length \
	--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen \
	--joined-dictionary \
	--destdir $</prep \
	--dataset-impl raw \
	--tokenizer space \
	--bpe sentencepiece \
	--workers 20 \
	--wandb-project tender_hamilton

# TRAINING
MINI_TRAIN_LSTM_SRCL: /srv/scratch2/kew/fairseq_materials/rrgen_012021/en/mini/
	mkdir -p $</lstm_srcl_bpemb200_hd200
	CUDA_VISIBLE_DEVICES=0 \
	fairseq-train \
	$</prep \
	-s review -t response \
	--arch rrgen_lstm_arch \
	--task rrgen_translation \
	--truncate-source --truncate-target \
	--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
	--dataset-impl raw \
	--max-epoch 10 \
	--max-tokens 10240 --update-freq 1 \
	--lr 0.001 --optimizer adam --clip-norm 0.8 \
	--encoder-embed-path /srv/scratch2/kew/embeddings/data/en/en.wiki.bpe.vs10000.d200.w2v.txt \
	--decoder-embed-path /srv/scratch2/kew/embeddings/data/en/en.wiki.bpe.vs10000.d200.w2v.txt \
	--encoder-embed-dim 200 --decoder-embed-dim 200 --decoder-out-embed-dim 200 \
	--share-all-embeddings \
	--encoder-hidden-size 200 \
	--decoder-hidden-size 200 \
	--use-sentiment alpha_sentiment \
	--use-category domain \
	--use-rating rating \
	--use-length review_length \
	--save-dir $</lstm_srcl_bpemb200_hd200/checkpoints/ --save-interval 4 \
	--wandb-project tender_hamilton

# DECODING
MINI_DECODE: /srv/scratch2/kew/fairseq_materials/rrgen_012021/en/mini/
	CUDA_VISIBLE_DEVICES=0 \
	fairseq-generate \
	$</prep \
	--path $</lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
	-s review -t response \
	--task rrgen_translation --truncate-source --truncate-target \
	--dataset-impl raw \
	--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
	--batch-size 10 \
	--sampling \
	--sampling-topk 5 \
	--nbest 5 \
	--remove-bpe sentencepiece \
	--use-sentiment alpha_sentiment \
	--use-category domain \
	--use-rating rating \
	--use-length review_length \
	--wandb-project tender_hamilton
	
# ---------------------------------------------

EN_UNFILTERED_PREP: /srv/scratch2/kew/fairseq_materials/rrgen_012021/en/no_filter
	fairseq-preprocess \
	--task rrgen_translation \
	--trainpref $</bpe/train \
	--validpref $</bpe/valid \
	--testpref $</bpe/test \
	--source-lang review \
	--target-lang response \
	--sent-ext alpha_sentiment \
	--rate-ext rating \
	--cate-ext domain \
	--len-ext review_length \
	--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen \
	--joined-dictionary \
	--destdir $</prep \
	--dataset-impl raw \
	--tokenizer space \
	--bpe sentencepiece \
	--workers 20 \
	--wandb-project readvisor

EN_UNFILTERED_TRAIN_LSTM_SRCL: /srv/scratch2/kew/fairseq_materials/rrgen_012021/en/no_filter
	mkdir -p $</lstm_srcl_bpemb200_hd200
	CUDA_VISIBLE_DEVICES=0 \
	nohup \
	fairseq-train \
	$</prep \
	-s review -t response \
	--arch rrgen_lstm_arch \
	--task rrgen_translation \
	--truncate-source --truncate-target \
	--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
	--dataset-impl raw \
	--max-epoch 20 \
	--max-tokens 10240 --update-freq 1 \
	--lr 0.001 --optimizer adam \
	--encoder-embed-path /srv/scratch2/kew/embeddings/data/en/en.wiki.bpe.vs10000.d200.w2v.txt \
	--decoder-embed-path /srv/scratch2/kew/embeddings/data/en/en.wiki.bpe.vs10000.d200.w2v.txt \
	--encoder-embed-dim 200 --decoder-embed-dim 200 --decoder-out-embed-dim 200 \
	--share-all-embeddings \
	--encoder-hidden-size 200 \
	--decoder-hidden-size 200 \
	--use-sentiment alpha_sentiment \
	--use-category domain \
	--use-rating rating \
	--use-length review_length \
	--wandb-project readvisor \
	--save-dir $</lstm_srcl_bpemb200_hd200/checkpoints/ --save-interval 4 >| $</lstm_srcl_bpemb200_hd200/train.log &
	
EN_UNFILTERED_DECODE_LSTM_SRCL_TOPK: /srv/scratch2/kew/fairseq_materials/rrgen_012021/en/no_filter
	CUDA_VISIBLE_DEVICES=0 \
	nohup \
	fairseq-generate \
	$</prep \
	--path $</lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
	-s review -t response \
	--task rrgen_translation --truncate-source --truncate-target \
	--dataset-impl raw \
	--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
	--batch-size 128 \
	--sampling \
	--sampling-topk 5 \
	--nbest 5 \
	--remove-bpe sentencepiece \
	--use-sentiment alpha_sentiment \
	--use-category domain \
	--use-rating rating \
	--use-length review_length \
	--wandb-project readvisor >| $</lstm_srcl_bpemb200_hd200/nbest5_topk5.txt &
	
# ---------------------------------------------
# SOUNDS GEN <= 0.7, SENT LEN >= 3, LR <=1.8, REx5
# ---------------------------------------------

EN_SG0.7_SL3_LR1.8_UP5_PREP: /srv/scratch2/kew/fairseq_materials/rrgen_012021/en/SG0.7_SL3_LR1.8_UP5
	fairseq-preprocess \
	--task rrgen_translation \
	--trainpref $</bpe/train \
	--validpref $</bpe/valid \
	--testpref $</bpe/test \
	--source-lang review \
	--target-lang response \
	--sent-ext alpha_sentiment \
	--rate-ext rating \
	--cate-ext domain \
	--len-ext review_length \
	--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen \
	--joined-dictionary \
	--destdir $</prep \
	--dataset-impl raw \
	--tokenizer space \
	--bpe sentencepiece \
	--workers 20 \
	--wandb-project readvisor

EN_SG0.7_SL3_LR1.8_UP5_TRAIN_LSTM_SRCL: /srv/scratch2/kew/fairseq_materials/rrgen_012021/en/SG0.7_SL3_LR1.8_UP5
	mkdir -p $</lstm_srcl_bpemb200_hd200
	CUDA_VISIBLE_DEVICES=1 \
	nohup \
	fairseq-train \
	$</prep \
	-s review -t response \
	--arch rrgen_lstm_arch \
	--task rrgen_translation \
	--truncate-source --truncate-target \
	--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
	--dataset-impl raw \
	--max-epoch 20 \
	--max-tokens 20480 --update-freq 1 \
	--lr 0.001 --optimizer adam --clip-norm 1.0 \
	--encoder-embed-path /srv/scratch2/kew/embeddings/data/en/en.wiki.bpe.vs10000.d200.w2v.txt \
	--decoder-embed-path /srv/scratch2/kew/embeddings/data/en/en.wiki.bpe.vs10000.d200.w2v.txt \
	--encoder-embed-dim 200 --decoder-embed-dim 200 --decoder-out-embed-dim 200 \
	--share-all-embeddings \
	--encoder-hidden-size 200 \
	--decoder-hidden-size 200 \
	--use-sentiment alpha_sentiment \
	--use-category domain \
	--use-rating rating \
	--use-length review_length \
	--wandb-project readvisor \
	--save-dir $</lstm_srcl_bpemb200_hd200/checkpoints/ --save-interval 4 >| $</lstm_srcl_bpemb200_hd200/train.log &
	
EN_SG0.7_SL3_LR1.8_UP5_DECODE_LSTM_SRCL_TOPK: /srv/scratch2/kew/fairseq_materials/rrgen_012021/en/SG0.7_SL3_LR1.8_UP5
	CUDA_VISIBLE_DEVICES=1 \
	nohup \
	fairseq-generate \
	$</prep \
	--path $</lstm_srcl_bpemb200_hd200/checkpoints/checkpoint_best.pt \
	-s review -t response \
	--task rrgen_translation --truncate-source --truncate-target \
	--dataset-impl raw \
	--user-dir /home/user/kew/INSTALLS/ffairseq/examples/rrgen/ \
	--batch-size 128 \
	--sampling \
	--sampling-topk 5 \
	--nbest 5 \
	--remove-bpe sentencepiece \
	--use-sentiment alpha_sentiment \
	--use-category domain \
	--use-rating rating \
	--use-length review_length \
	--wandb-project readvisor >| $</lstm_srcl_bpemb200_hd200/nbest5_topk5.txt &
