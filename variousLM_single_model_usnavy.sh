start=`date +%s`

declare -a lrlist=('ctrl' 't5-large' 'xlm-mlm-en-2048' 'xlm-roberta-large' 'facebook/bart-large-cnn' 'DialoGPT-large')
for i in "${lrlist[@]}"; do python3 ./model_train.py --dataset=usnavy --k_folds=5 --test_run="$i" --fine_tune; done
for i in "${lrlist[@]}"; do for j in {0..4}; do python3 embed_tweets.py --dataset=usnavy --fold=$j --test_run="$i"; done; done
for i in "${lrlist[@]}"; do python3 ./ml_single_model_procedure.py --test_run="$i" --dataset=usnavy;done

end=`date +%s`

runtime=$((end-start))
echo $runtime
destdir=./roberta_optim/bert_optim_time_usnavy.txt
echo $runtime > $destdir