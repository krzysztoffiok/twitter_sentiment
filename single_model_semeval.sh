start=`date +%s`

declare -a lrlist=('microsoft/DialoGPT-large')
declare -a secondarylist=('microsoft/DialoGPT-large' 'facebook/bart-large-cnn' 'xlm-mlm-en-2048' 'xlm-roberta-large')
declare -a thirdlist=('microsoft/DialoGPT-large' 'facebook/bart-large-cnn' 'xlm-mlm-en-2048' 'xlm-roberta-large' 'bert-large-cased' 'bert-large-uncased')
python3 semeval_data_splitter_optim.py
for i in "${lrlist[@]}"; do python3 ./model_train.py --dataset=semeval --k_folds=5 --test_run="$i" --fine_tune; done
python3 semeval_data_splitter.py
for i in "${lrlist[@]}"; do for j in {0..4}; do python3 embed_tweets.py --dataset=semeval --fold=$j --test_run="$i" --subset=train; done; done
for i in "${lrlist[@]}"; do for j in {0..4}; do python3 embed_tweets.py --dataset=semeval --fold=$j --test_run="$i" --subset=test; done; done
for i in "${lrlist[@]}"; do python3 ./ml_single_model_procedure.py --test_run="$i" --dataset=semeval

end=`date +%s`

runtime=$((end-start))
echo $runtime
destdir=./roberta_optim/dialogpt_optim_time_semeval.txt
echo $runtime > $destdir