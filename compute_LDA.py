import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import random
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Classify data')
parser.add_argument('--fold', required=False, type=int, default=5)
parser.add_argument('--dataset', required=False, type=str, default="usnavy",
                    help="dataset usnavy or semeval")

args = parser.parse_args()
_fold = args.fold

dataset = args.dataset

if dataset == "usnavy":
    # <H1> Preparing data split: 5 fold cross validation </H1>
    # <h3> The procedure is identical as when splitting data for the purpose of training selected Language Models
    df = pd.read_excel(f"./{dataset}_data/source_data/tweet_sentiment_input_file.xlsx", converters={'dummy_id': str})

    # drop not needed columns
    df = df.drop(["row", "dummy_id"], axis=1)

    # 5 fold CV
    # setup random state
    np.random.seed(13)

    # define number of folds
    fold_number = 5
    kf = KFold(n_splits=fold_number, random_state=13, shuffle=True)

    # create data splits for Deep Learning Language Models trained with Flair framework
    train_indexes = {}
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
        i += 1

    # test sets for Machine Learning are equal to those for Flair framework
    test_ml = test_indexes
    train_ml = train_ml

    # the below code is mostly inspired by https://github.com/kapadias/mediumposts/blob/master/nlp/published_notebooks/Evaluate%20Topic%20Models.ipynb
    # and https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/
    import gensim
    from gensim.utils import simple_preprocess
    import nltk
    from nltk.corpus import stopwords
    import spacy
    import gensim.corpora as corpora
    nltk.download('stopwords')
    # Initialize spacy 'en' model, keeping only tagger component (for efficiency)
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

    stop_words = stopwords.words('english')
    stop_words.extend(['from', 'subject', 're', 'edu', 'use'])

    def sent_to_words(sentences):
        for sentence in sentences:
            yield gensim.utils.simple_preprocess(str(sentence), deacc=True)

    # Define functions for stopwords, bigrams, trigrams and lemmatization
    def remove_stopwords(texts, stop_words=stop_words):
        return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

    def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """https://spacy.io/api/annotation"""
        texts_out = []
        for sent in texts:
            doc = nlp(" ".join(sent))
            texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
        return texts_out

    def format_topics_sentences(ldamodel, corpus):
        topic_list = list()
        for i, row in enumerate(ldamodel[corpus]):
            row = sorted(row, key=lambda x: (x[1]), reverse=True)
            topic_list.append(row[0][0])
        return topic_list

    def create_corpus(idx, stop_words=stop_words):
        docs = df['pretext'].iloc[idx].to_list()
        data_words = list(sent_to_words(docs))
        # Build the bigram and trigram models
        bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)  # higher threshold fewer phrases.
        trigram = gensim.models.Phrases(bigram[data_words], threshold=100)

        # Faster way to get a sentence clubbed as a trigram/bigram
        bigram_mod = gensim.models.phrases.Phraser(bigram)
        # trigram_mod = gensim.models.phrases.Phraser(trigram)
        # Remove Stop Words
        data_words_nostops = remove_stopwords(data_words)
        # Form Bigrams
        data_words_bigrams = [bigram_mod[doc] for doc in data_words_nostops]
        # Do lemmatization keeping only noun, adj, vb, adv
        data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
        # Create Dictionary
        id2word = corpora.Dictionary(data_lemmatized)
        # Create Corpus
        texts = data_lemmatized
        # Term Document Frequency
        corpus = [id2word.doc2bow(text) for text in texts]
        return corpus, id2word

    df['pretext'] = df['text'].map(lambda x: x.lower())

    whole_corpus, id2word = create_corpus([x for x in range(len(df['pretext']))])

    # Build LDA model
    for n_topics in [5, 8, 11]:
        lda_model = gensim.models.LdaMulticore(corpus=whole_corpus,
                                               id2word=id2word,
                                               num_topics=n_topics,
                                               random_state=100,
                                               chunksize=100,
                                               passes=10,
                                               alpha='asymmetric',
                                               eta=0.91
                                               )

        topic_list = format_topics_sentences(ldamodel=lda_model, corpus=whole_corpus)

        df[f"{n_topics}_topic"] = topic_list

    df.to_excel(f"./{dataset}_data/source_data/LDA.xlsx")


