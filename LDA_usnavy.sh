start=`date +%s`

declare -a lrlist=('roberta_ft')

do python3 ./compute_LDA.py

# only LDA topics are used
for i in "${lrlist[@]}"; do for j in {0..4}; do python3 ./ml_single_model_procedure_LDA.py --test_run="$i" --estimators=250 \
 --fold=$j --topic --only_topic; done; done

# transformer embeddings and LDA topics are used
 for i in "${lrlist[@]}"; do for j in {0..4}; do python3 ./ml_single_model_procedure_LDA.py --test_run="$i" --estimators=250 \
 --fold=$j --topic; done; done

# only transformer embeddings are used
 for i in "${lrlist[@]}"; do for j in {0..4}; do python3 ./ml_single_model_procedure_LDA.py --test_run="$i" --estimators=250 \
 --fold=$j; done; done


end=`date +%s`

runtime=$((end-start))
echo $runtime
destdir=./LDA_USNavy.txt
echo $runtime > $destdir

