import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os

"""
Splits the data into 5 folds for Flair model training and later ML
"""

# read original data file
df = pd.read_excel("./usnavy_data/source_data/tweet_sentiment_input_file.xlsx", converters={'dummy_id': str})

# drop not needed columns
df = df.drop(["row", "dummy_id"], axis=1)
# change format of 'sentiment' label for further training in Flair framework
df['sentiment'] = '__label__' + df['sentiment'].astype(str)

# 5 fold CV
# setup random state and folds
np.random.seed(13)
fold_number = 5
kf = KFold(n_splits=fold_number, random_state=13, shuffle=True)

# create data splits for Deep Learning Language Models trained with Flair framework
train_indexes = {}
val_indexes = {}
test_indexes = {}

# train sets for Machine Learning
train_ml = {}
i = 0

# this split (with fold_number=5) results in: 20% test, 10% val, 70% train for Flair framework
# and the same 20% test and 80 % train for Machine Learning
indexes = list(range(0, len(df)))
for train_index, test_index in kf.split(indexes):
    test_indexes[i] = test_index
    train_ml[i] = train_index
    train_index, val_index = train_test_split(train_index, test_size=0.125, random_state=13, shuffle=True)
    train_indexes[i] = train_index
    val_indexes[i] = val_index
    i += 1
    
# test sets for Machine Learning are equal to those for Flair framework
test_ml = test_indexes

# create folders for FLAIR data splits and .tsv files for training
folds_path1 = []
for fold in range(fold_number):
    folds_path1.append('./usnavy_data/model_sentiment_{}/'.format(str(fold)))
    try:
        os.mkdir('./usnavy_data/model_sentiment_{}'.format(str(fold)))
    except FileExistsError:
        None  # continue
    df.iloc[test_indexes[fold]].to_csv(os.path.join(folds_path1[fold], "test_.tsv"),
                                       index=False, header=False, encoding='utf-8', sep='\t')
    df.iloc[train_indexes[fold]].to_csv(os.path.join(folds_path1[fold], "train.tsv"),
                                        index=False, header=False, encoding='utf-8', sep='\t')
    df.iloc[val_indexes[fold]].to_csv(os.path.join(folds_path1[fold], "dev.tsv"),
                                      index=False, header=False, encoding='utf-8', sep='\t')
