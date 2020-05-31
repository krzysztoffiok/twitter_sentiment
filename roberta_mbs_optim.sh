start=`date +%s`

declare -a mbslist=('04' '08' '12' '16' '32')
python3 ./roberta_mbs_optim/semeval_data_splitter_optim.py
for i in "${mbslist[@]}"; do python3 ./roberta_mbs_optim/model_train_optim.py --dataset=semeval --k_folds=5 --test_run=roberta-large"$i" --fine_tune; done
python3 semeval_data_splitter.py
for i in "${mbslist[@]}"; do for j in {0..4}; do python3 embed_tweets.py --dataset=semeval --fold=$j --test_run=roberta-large"$i" --subset=train; done; done
for i in "${mbslist[@]}"; do for j in {0..4}; do python3 embed_tweets.py --dataset=semeval --fold=$j --test_run=roberta-large"$i" --subset=test; done; done

end=`date +%s`

runtime=$((end-start))
echo $runtime
destdir=./roberta_mbs_optim/roberta_bs_fold_optim_time.txt
echo $runtime > $destdir