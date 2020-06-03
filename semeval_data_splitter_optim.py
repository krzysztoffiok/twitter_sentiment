import pandas as pd
import numpy as np
import os
import datatable as dt

df_semtest = dt.fread("./semeval_data/source_data/semtest_filtered.csv").to_pandas()
df_semtrain = dt.fread("./semeval_data/source_data/semtrain_filtered.csv").to_pandas()
df_semtest.drop("C0", axis=1, inplace=True)
df_semtrain.drop("C0", axis=1, inplace=True)

# 5 fold CV
# setup random state
np.random.seed(13)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
fold_number = 5
kf = KFold(n_splits=fold_number, random_state=13, shuffle=True)

# create data splits for Deep Learning Language Models trained with Flair framework
train_indexes = {}
val_indexes = {}

# train sets for Machine Learning
train_ml = {}
i = 0

# this split (with fold_number=5) results in: 20% val, 80% train for Flair framework
indexes = list(range(0, len(df_semtrain)))
for train_index, val_index in kf.split(indexes):
    val_indexes[i] = val_index
    train_indexes[i] = train_index
    i += 1

# add string required by Flair framework
df_semtrain['sentiment'] = '__label__' + df_semtrain['sentiment'].astype(str)
df_semtest['sentiment'] = '__label__' + df_semtest['sentiment'].astype(str)

# create folders for FLAIR data splits and .tsv files for training
folds_path1 = []
for fold in range(fold_number):
    folds_path1.append('./semeval_data/model_sentiment_{}/'.format(str(fold)))
    try:
        os.mkdir('./semeval_data/model_sentiment_{}'.format(str(fold)))
    except FileExistsError:
        None  # continue

    # df_semtest.to_csv(os.path.join(folds_path1[fold], "test_.tsv"), index=False, header=False, encoding='utf-8', sep='\t')
    df_semtrain.iloc[val_indexes[fold]].to_csv(os.path.join(folds_path1[fold], "test_.tsv"), index=False, header=False, encoding='utf-8',
                      sep='\t')
    df_semtrain.iloc[train_indexes[fold]].to_csv(os.path.join(folds_path1[fold], "train.tsv"), index=False, header=False, encoding='utf-8', sep='\t')
    df_semtrain.iloc[val_indexes[fold]].to_csv(os.path.join(folds_path1[fold], "dev.tsv"), index=False, header=False, encoding='utf-8', sep='\t')

