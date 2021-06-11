#! /bin/bash

scripts=`dirname "$0"`
base=$scripts/..

data=$base/data

mkdir -p $data

wget http://data.statmt.org/wmt17/translation-task/training-parallel-nc-v12.tgz -P $data
tar -xzvf $data/training-parallel-nc-v12.tgz -C $data

head -n 10000 $data/training/news-commentary-v12.de-en.de > $data/train.de
head -n 10000 $data/training/news-commentary-v12.de-en.en > $data/train.en

head -n 12000 $data/training/news-commentary-v12.de-en.de | tail -n 2000 > $data/dev.de
head -n 12000 $data/training/news-commentary-v12.de-en.en | tail -n 2000 > $data/dev.en

head -n 14000 $data/training/news-commentary-v12.de-en.de | tail -n 2000 > $data/test.de
head -n 14000 $data/training/news-commentary-v12.de-en.en | tail -n 2000 > $data/test.en

# sizes
echo "Sizes of corpora:"
for corpus in train dev test; do
	echo "corpus: "$corpus
	wc -l $data/$corpus.de $data/$corpus.en
done

# sanity checks
echo "At this point, please make sure that 1) number of lines are as expected, 2) language suffixes are correct and 3) files are parallel"
