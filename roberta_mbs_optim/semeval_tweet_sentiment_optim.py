import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, GaussianNB
import sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import xgboost as xgb
import random
import shap
import datatable as dt
import argparse
import _utils
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Classify data')
parser.add_argument('--fold', required=False, type=int, default=0)
parser.add_argument('--shap', required=False, default=None, help='If argument is passed, SHAP explanations for'
                                                                 ' a selected LM will be computed. Possible values'
                                                                 'include e.g. SEANCE, LIWC, Term Frequency')
parser.add_argument('--samples', required=False, type=int, default=200,
                    help="number of samples of data to explain by SHAP")
parser.add_argument('--estimators', required=False, type=int, default=250,
                    help="number of estimators in machine learning classification models")

args = parser.parse_args()
_fold = args.fold
_shap = args.shap
_estimators = args.estimators
_shap_samples = args.samples

# choose the dataset
dataset = "semeval"

# <H1> Preparing data split: 5 fold cross validation </H1>
# <h3> The procedure is identical as when splitting data for the purpose of training selected Language Models

# read original data file
if dataset == "semeval":
    df_semtest = dt.fread("./semeval_data/source_data/semtest.csv").to_pandas()
    df_semtrain = dt.fread("./semeval_data/source_data/semtrain.csv").to_pandas()  
    df = pd.concat([df_semtrain, df_semtest], ignore_index=True)

# define number of folds
fold_number = 5

# create data splits for Deep Learning Language Models trained with Flair framework
train_indexes = {}
test_indexes = {}

# train sets for Machine Learning
train_ml = {}
test_ml = {}
i = 0

# Since in SEMEVAL only a single train and test split were given, the five fold cross validation
# is carried out only when training split i divided into train and val for Flair and training
# of DL models. Here, we repeat 5 times the same train and test splits, however DL models for each
# fold were previously separately trained.
indexes = list(range(0, len(df)))
for fold in range(fold_number):
    # first 6000 rows are training instances, rest is for testing. The difference in folds comes from
    # different train/val splits when DL trainable models were concerned
    test_ml[i] = df[6000:].index
    train_ml[i] = df[:6000].index
    i += 1

# <h3> Vector representations (embeddings) created by selected Deep Learning Language Models trained previously
# on here addressed task
# define which embedding files to read
bslist = ["04", "08", "12", "16", "32"]
embeddings = [(x, f"roberta-large{x}") for x in bslist]

# instantiate list of data frames with features and a list of feature names for each df
dfemblist = []

# Initialize a dictionary with all features used later on in Machine Learning
allFeatures = {}

# read embedding files and define corresponding feature names (lists of names)
for emname, embedding in embeddings:
    embfeaturedict = {}
    for fold in range(fold_number):
        # read encoded sentences by the selected language model
        train = dt.fread(f"./{dataset}_data/embeddings/{embedding}_encoded_sentences_{fold}_train.csv").\
            to_pandas()
        test = dt.fread(f"./{dataset}_data/embeddings/{embedding}_encoded_sentences_{fold}_test.csv").\
            to_pandas()
        dfemb = pd.concat([train, test], ignore_index=True)

        embfeatures = [f"{emname}{fold}row"]

        # define number of feature columns (columns - 3)
        number_of_feature_columns = len(dfemb.columns) - 3

        # create unique feature (column) names
        embfeatures.extend([f"{emname}{fold}{x}" for x in range(number_of_feature_columns)])
        embfeatures.extend([f"{emname}{fold}_sentiment_", f"{emname}{fold}_dummy_id_"])
        dfemb.columns = embfeatures

        # append features from each language model in tuple ((model_name,fold), [features])
        embfeaturedict[fold] = [f"{emname}{fold}{x}" for x in range(number_of_feature_columns)]

        # append encoded sentences by the selected language model to a list of data frames
        dfemblist.append(dfemb)

    # create entry in dictionary with all features for each trained language model
    allFeatures[emname] = embfeaturedict

# concat all Data Frames: liwc, TF, DL embedding into one df_ml that will be used in Machine Learning
for dfemb in dfemblist:
    df_ml = pd.concat([df_ml, dfemb], axis=1)

# define the target variable in the final df_ml data frame
df_ml["target_ml"] = df["sentiment"]

# <h1> Machine Learning part
# Define list of model names
all_language_models = [x for x in bslist]

if _shap is None:

    # instantiate dictionary for data frames with results
    allPreds = {}
    allTrues = {}

    # define which classification models to use
    models = [xgb.XGBClassifier(objective='multi:softprob', n_jobs=24, learning_rate=0.03,
                                max_depth=10, subsample=0.7, colsample_bytree=0.6,
                                random_state=2020, n_estimators=_estimators, tree_method='gpu_hist')]

    # use features from selected language models
    for language_model in all_language_models:
        # for training of selected classification models
        for classification_model in models:
            preds, trues = _utils.ML_classification(allFeatures=allFeatures, train_ml=train_ml, test_ml=test_ml,
                                                    df_ml=df_ml, classification_model=classification_model,
                                                    language_model=language_model, fold_number=fold_number)

            # save model predictions
            allPreds[f"{language_model}_{type(classification_model).__name__}"] = preds.copy()
            allTrues[f"{language_model}_{type(classification_model).__name__}"] = trues.copy()

    # save model predictions together with true sentiment labels
    pd.DataFrame(allPreds).to_csv(f"./{dataset}_data/predictions.csv")
    pd.DataFrame(allTrues).to_csv(f"./{dataset}_data/trues.csv")

    _utils.compute_metrics(dataset=dataset)

else:

    # prepare model for SHAP explanations
    shap_model, train_data, test_data = _utils.train_model_for_shap(allFeatures=allFeatures, train_ml=train_ml,
                                                                    test_ml=test_ml, df_ml=df_ml, classification_model=
                                                                    xgb.XGBClassifier(objective='multi:softprob',
                                                                                      n_jobs=24, learning_rate=0.03,
                                                                                      max_depth=10, subsample=0.7,
                                                                                      colsample_bytree=0.6,
                                                                                      random_state=2020,
                                                                                      n_estimators=_estimators,
                                                                                      tree_method='gpu_hist'),
                                                                    language_model=_shap, fold=_fold)

    fig = _utils.explain_model(model=shap_model, train_data=train_data, test_data=test_data, samples=_shap_samples)
    plt.savefig(f'./results/{dataset}_{_shap}_{_fold}_{_estimators}_summary_plot.png')

