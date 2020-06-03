start=`date +%s`

declare -a lrlist=('bert-large-cased' 'bert-large-uncased')
for i in "${lrlist[@]}"; do python3 ./model_train.py --dataset=semeval --k_folds=5 --test_run="$i" --fine_tune; done
for i in "${lrlist[@]}"; do for j in {0..4}; do python3 embed_tweets.py --dataset=semeval --fold=$j --test_run="$i" --subset=train; done; done
for i in "${lrlist[@]}"; do for j in {0..4}; do python3 embed_tweets.py --dataset=semeval --fold=$j --test_run="$i" --subset=test; done; done
for i in "${lrlist[@]}"; do python3 ./ml_single_model_procedure.py --test_run="$i" --dataset=semeval

end=`date +%s`

runtime=$((end-start))
echo $runtime
destdir=./roberta_optim/bert_optim_time_semeval.txt
echo $runtime > $destdir