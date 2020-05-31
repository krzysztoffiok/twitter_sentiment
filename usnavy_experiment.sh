start=`date +%s`

python3 usnavy_data_splitter.py
python3 model_train.py --dataset=usnavy --k_folds=5 --test_run=fasttext
python3 model_train.py --dataset=usnavy --k_folds=5 --test_run=roberta_lstm
python3 model_train.py --dataset=usnavy --k_folds=5 --test_run=roberta_ft --fine_tune
for i in {0..4}; do python3 embed_tweets.py --dataset=usnavy --fold=$i --test_run=roberta_ft; done
for i in {0..4}; do python3 embed_tweets.py --dataset=usnavy --fold=$i --test_run=fasttext; done
for i in {0..4}; do python3 embed_tweets.py --dataset=usnavy --fold=$i --test_run=roberta_lstm; done
python3 embed_tweets.py --dataset=usnavy --pool=True  --test_run=fasttext
python3 embed_tweets.py --dataset=usnavy --pool=True  --test_run=roberta
python3 embed_tweets.py --dataset=usnavy --use=True
python3 usnavy_tweet_sentiment.py

end=`date +%s`

runtime=$((end-start))
echo $runtime
destdir=./usnavy_experiment_time.txt
echo $runtime > $destdir