start=`date +%s`

python3 semeval_data_convert_to_csv.py
python3 semeval_filter.py
python3 ./roberta_mbs_optim/semeval_data_splitter_optim.py
python3 model_train.py --dataset=semeval --k_folds=5 --test_run=fasttext
python3 model_train.py --dataset=semeval --k_folds=5 --test_run=roberta_lstm
python3 model_train.py --dataset=semeval --k_folds=5 --test_run=roberta_ft --fine_tune
python3 semeval_data_splitter.py
for i in {0..4}; do python3 embed_tweets.py --dataset=semeval --fold=$i --test_run=roberta_ft --subset=train; done
for i in {0..4}; do python3 embed_tweets.py --dataset=semeval --fold=$i --test_run=roberta_ft --subset=test; done
for i in {0..4}; do python3 embed_tweets.py --dataset=semeval --fold=$i --test_run=fasttext --subset=train; done
for i in {0..4}; do python3 embed_tweets.py --dataset=semeval --fold=$i --test_run=fasttext --subset=test; done
for i in {0..4}; do python3 embed_tweets.py --dataset=semeval --fold=$i --test_run=roberta_lstm --subset=train; done
for i in {0..4}; do python3 embed_tweets.py --dataset=semeval --fold=$i --test_run=roberta_lstm --subset=test; done
python3 embed_tweets.py --dataset=semeval --pool=True  --test_run=fasttext --subset=train
python3 embed_tweets.py --dataset=semeval --pool=True  --test_run=fasttext --subset=test
python3 embed_tweets.py --dataset=semeval --pool=True  --test_run=roberta --subset=train
python3 embed_tweets.py --dataset=semeval --pool=True  --test_run=roberta --subset=test
python3 embed_tweets.py --dataset=semeval --use=True --subset=train
python3 embed_tweets.py --dataset=semeval --use=True --subset=test
python3 semeval_tweet_sentiment.py

end=`date +%s`

runtime=$((end-start))
echo $runtime
destdir=./semeval_experiment_time.txt
echo $runtime > $destdir