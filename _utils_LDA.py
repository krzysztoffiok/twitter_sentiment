from sklearn import metrics
import shap
import random
from sklearn import metrics
from scipy.stats import wasserstein_distance
import datatable as dt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif


def train_model_for_shap(allFeatures, train_ml, test_ml, df_ml, classification_model, language_model, fold):
    """
    Function to train a single Language Model for SHAP explanations
    Example use: classification_model=RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=2,
                             min_samples_leaf=1, max_features='auto', n_jobs=-1, random_state=2020),
                language_model="Term Frequency",
                fold = 2
    possible options for language model are: "Term Frequency" or "LIWC".
    possible fold values: 0, 1, 2, 3, 4
    """
    # list of analyzed language models
    model = classification_model
    print(type(model).__name__)

    features = set(allFeatures[language_model][fold])
    preds = []
    trues = []

    train_index = train_ml[fold]
    test_index = test_ml[fold]

    train_data = df_ml[features].iloc[train_index]
    target_train_data = df_ml["target_ml"].iloc[train_index]
    test_data = df_ml[features].iloc[test_index]
    target_test_data = df_ml.iloc[test_index]["target_ml"]
    model.fit(train_data, target_train_data)

    preds.append(model.predict(test_data).tolist())
    trues.append(target_test_data.tolist())

    print(language_model)
    mcc = metrics.matthews_corrcoef(y_true=sum(trues, []), y_pred=sum(preds, []))
    f1 = metrics.f1_score(y_true=sum(trues, []), y_pred=sum(preds, []), average="weighted")
    print("MCC: ", round(mcc, 3))
    print("F1: ", round(f1, 3))
    return model, train_data, test_data


def explain_model(model, train_data, test_data, samples):
    """
    Function that computes and displays SHAP model explanations
    """
    model_name = type(model).__name__
    random.seed(13)
    samples_to_explain = samples
    if model_name not in ["RandomForestClassifier", "XGBClassifier"]:
        explainer = shap.KernelExplainer(model.predict_proba, train_data[:50], link="identity")
        shap_values = explainer.shap_values(train_data[:50], nsamples=200, l1_reg="num_features(100)")

    else:
        explainer = shap.TreeExplainer(model, data=shap.sample(train_data, samples_to_explain),
                                       feature_perturbation='interventional')
        shap_values = explainer.shap_values(shap.sample(train_data, samples_to_explain), check_additivity=False)

    fig = shap.summary_plot(shap_values, test_data, max_display=5, show=False)
    return fig


def ML_classification(allFeatures, train_ml, test_ml, df_ml, classification_model, language_model, fold_number, topic,
                      n_comp_list, only_topic, term_frequency):
    """
    Function to train classification models on features provided by language models
    Example use: classification_model=RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=2,
                             min_samples_leaf=1, max_features='auto', n_jobs=-1, random_state=2020)
                language_model=
    possible options for language model list are: "Term Frequency", "LIWC", "Pooled FastText", "Pooled RoBERTa"
     or "Universal Sentence Encoder"
    """
    # list of analyzed language models
    model = classification_model
    preds = []
    trues = []
    print("Now training: ", language_model, " ", type(model).__name__)
    # for each fold
    if term_frequency:
        language_model = "Term Frequency"

    for fold in range(fold_number):
        # chose appropriate features and data
        features = list(set(allFeatures[language_model][fold]))
        if topic:
            for n in n_comp_list:
                features.append(f'{n}_topic')
                if only_topic:
                    features = [f'{n}_topic']

        train_index = train_ml[fold]
        test_index = test_ml[fold]

        train_data = df_ml[features].iloc[train_index]
        target_train_data = df_ml["target_ml"].iloc[train_index]
        test_data = df_ml[features].iloc[test_index]
        target_test_data = df_ml.iloc[test_index]["target_ml"]
        model.fit(train_data, target_train_data)

        preds.append(model.predict(test_data).tolist())
        trues.append(target_test_data.tolist())
    return sum(preds, []), sum(trues, [])


def compute_metrics(dataset, test_run, topic, only_topic, term_frequency):
    # read result files to compute metrics including SemEval2017-specific
    preds = dt.fread(f"./{dataset}_data/{topic}_{only_topic}_{test_run}_predictions.csv").to_pandas()
    trues = dt.fread(f"./{dataset}_data/{topic}_{only_topic}_{test_run}_trues.csv").to_pandas()
    if term_frequency:
        preds = dt.fread(f"./{dataset}_data/{term_frequency}_{topic}_{only_topic}_{test_run}_predictions.csv").to_pandas()
        trues = dt.fread(f"./{dataset}_data/{term_frequency}_{topic}_{only_topic}_{test_run}_trues.csv").to_pandas()

    modelColNames = preds.columns.to_list()
    modelColNames.remove("C0")

    # define classes and indexes of true values for each class. For each model the true index values are the
    # same since the test set was the same.
    classes = set(trues[f"{modelColNames[0]}"])
    cls_index = dict()
    for cls in classes:
        cls_index[cls] = trues[trues[f"{modelColNames[0]}"] == cls].index.to_list()

    # for each model compute the metrics
    allmetrics = dict()
    for model in modelColNames:
        model_metrics = dict()
        mae = metrics.mean_absolute_error(y_true=trues[f"{model}"], y_pred=preds[f"{model}"])
        emd = wasserstein_distance(trues[f"{model}"], preds[f"{model}"])
        mcc = metrics.matthews_corrcoef(y_true=trues[f"{model}"], y_pred=preds[f"{model}"])
        f1 = metrics.f1_score(y_true=trues[f"{model}"], y_pred=preds[f"{model}"], average="macro")

        # class wise computation of mean absolute error and later averaging over classes to implement
        # MAEM from SemEval2017
        mae_dict = {}

        for cls in classes:
            local_trues = trues[f"{model}"].iloc[cls_index[cls]]
            local_preds = preds[f"{model}"].iloc[cls_index[cls]]
            mae_dict[cls] = metrics.mean_absolute_error(y_true=local_trues, y_pred=local_preds)

        mmae = np.array(list(mae_dict.values())).mean()
        _metrics = {"MMAE": mmae, "MAE": mae, "EMD": emd, "MCC": mcc, "F1": f1}
        for metric in _metrics.keys():
            model_metrics[metric] = _metrics[metric]

        allmetrics[model] = model_metrics

    dfmetrics = pd.DataFrame.from_dict(allmetrics)
    dfmetrics.to_csv(f"{term_frequency}_{topic}_{only_topic}_{dataset}_{test_run}_metric_results.csv")
    print(dfmetrics)


def compute_metrics_comb(dataset):
    # read result files to compute metrics including SemEval2017-specific
    preds = dt.fread(f"./{dataset}_data/_predictions.csv").to_pandas()
    trues = dt.fread(f"./{dataset}_data/_trues.csv").to_pandas()
    modelColNames = preds.columns.to_list()
    modelColNames.remove("C0")

    # define classes and indexes of true values for each class. For each model the true index values are the
    # same since the test set was the same.
    classes = set(trues[f"{modelColNames[0]}"])
    cls_index = dict()
    for cls in classes:
        cls_index[cls] = trues[trues[f"{modelColNames[0]}"] == cls].index.to_list()

    # for each model compute the metrics
    allmetrics = dict()
    for model in modelColNames:
        model_metrics = dict()
        mae = metrics.mean_absolute_error(y_true=trues[f"{model}"], y_pred=preds[f"{model}"])
        emd = wasserstein_distance(trues[f"{model}"], preds[f"{model}"])
        mcc = metrics.matthews_corrcoef(y_true=trues[f"{model}"], y_pred=preds[f"{model}"])
        f1 = metrics.f1_score(y_true=trues[f"{model}"], y_pred=preds[f"{model}"], average="macro")

        # class wise computation of mean absolute error and later averaging over classes to implement
        # MAEM from SemEval2017
        mae_dict = {}

        for cls in classes:
            local_trues = trues[f"{model}"].iloc[cls_index[cls]]
            local_preds = preds[f"{model}"].iloc[cls_index[cls]]
            mae_dict[cls] = metrics.mean_absolute_error(y_true=local_trues, y_pred=local_preds)

        mmae = np.array(list(mae_dict.values())).mean()
        _metrics = {"MMAE": mmae, "MAE": mae, "EMD": emd, "MCC": mcc, "F1": f1}
        for metric in _metrics.keys():
            model_metrics[metric] = _metrics[metric]

        allmetrics[model] = model_metrics

    dfmetrics = pd.DataFrame.from_dict(allmetrics)
    dfmetrics.to_csv(f"{dataset}_comb_metric_results.csv")
    print(dfmetrics)


def term_frequency(train_ml, df, allFeatures):
    foldTFfeatures = {}
    allWords = []
    for fold, rows in train_ml.items():
        vectorizer = CountVectorizer(min_df=4, binary=True)
        tf = vectorizer.fit_transform(df.iloc[rows]["text"])
        dftf = pd.DataFrame(tf.A, columns=vectorizer.get_feature_names())
        mi_imps = list(zip(mutual_info_classif(dftf, df.iloc[rows]["sentiment"], discrete_features=True), dftf.columns))
        mi_imps = sorted(mi_imps, reverse=True)
        topFeaturesN = 300
        foldTFfeatures[fold] = [f"TF_{y}" for x, y in mi_imps[0:topFeaturesN]].copy()
        # save all words found by TF models as important features
        allWords.extend([y for x, y in mi_imps[0:topFeaturesN]].copy())

    # add the Term Frequency language model key to dictionary with allFeatures from various language models
    allFeatures["Term Frequency"] = foldTFfeatures

    # Create TF features for all the text instances and create a corresponding data frame
    allWords = list(set(allWords))
    vectorizer = CountVectorizer(min_df=4, binary=True, vocabulary=allWords)
    tf = vectorizer.fit_transform(df["text"])
    dftf = pd.DataFrame(tf.A, columns=vectorizer.get_feature_names())
    dftf.columns = [f"TF_{x}" for x in dftf.columns]

    return dftf, allFeatures
