import pandas as pd
import argparse
import numpy as np
import flair
import torch
import datatable
import time
import math

""" example use to create tweet-level embeddings
# remember to start by setting a proper --dataset i.e. usnavy or semeval. Default is usnavy.
# If you choose dataset=semeval, it is required to add the --subset=train and --subset=test in another run.
python3 embed_tweets.py --dataset=semeval --fold=3 --test_run=fasttext --subset=test
Important: needs to be run separately for each fold e.g. with bash command:
for i in {0..4}; do python3 embed_tweets.py --dataset=semeval --fold=$i --test_run=fasttext  --subset=test; done

# with LSTM training over the pre-trained model
python3 embed_tweets.py --dataset=semeval --fold=3 --test_run=fasttext
python3 embed_tweets.py --fold=5 --test_run=roberta_lstm
Important: needs to be run separately for each fold e.g. with bash command:
for i in {0..4}; do python3 embed_tweets.py --fold=$i --test_run=fasttext; done

# no LSTM, just mean of pre-trained token embeddings without fine-tuning
python3 embed_tweets.py --pool=True  --test_run=fasttext
python3 embed_tweets.py --pool=True  --test_run=roberta

# CLS token output of fine-tuned transformer model
python3 embed_tweets.py --fold=0 --test_run=roberta_ft
Important: needs to be run separately for each fold e.g. with bash command:
for i in {0..4}; do python3 embed_tweets.py --fold=$i --test_run=roberta_ft; done

# universal sentence encoder
python3 embed_tweets.py --use=True

"""
flair.device = torch.device('cuda')

parser = argparse.ArgumentParser(description='Classify data')
parser.add_argument('--test_run', required=False, default='',
                    type=str, help='name of model')
parser.add_argument("--nrows", required=False, default=40000, type=int)
parser.add_argument("--fold", required=False, default=5, type=int)
parser.add_argument("--pool", required=False, default=False, type=bool)
parser.add_argument("--use", required=False, default=False, type=bool)
parser.add_argument('--dataset', required=False, type=str, default='usnavy')
parser.add_argument('--subset', required=False, type=str, default='train')
parser.add_argument("--bs", required=False, default=32, type=int)
args = parser.parse_args()

fold = args.fold
test_run = args.test_run
if "/" in test_run:
    test_run = test_run.split("/")[1]
nrows = args.nrows
pool = args.pool
_use = args.use
dataset = args.dataset
if dataset == "usnavy":
    subset = ""
else:
    subset = args.subset
bs = args.bs

# read data
if dataset == "usnavy":
    df = pd.read_excel(f"./{dataset}_data/source_data/tweet_sentiment_input_file.xlsx", converters={'dummy_id': str})
elif dataset == "semeval":
    df = datatable.fread(f"./{dataset}_data/source_data/sem{subset}_filtered.csv").to_pandas()
    df["dummy_id"] = [x for x in range(len(df))]
print(len(df))
df = df.head(nrows)
data = df.copy()

data = data[['text', 'sentiment', "dummy_id"]]

# if not universal sentence encoder
if not _use:
    # load Flair
    import torch
    import flair
    from flair.models import TextClassifier

    # load various embeddings
    from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, RoBERTaEmbeddings
    from flair.data import Sentence

    # if trained embeddings
    if not pool:
        # embeddings trained to the "downstream task"
        model = TextClassifier.load(f'./{dataset}_data/model_sentiment_{fold}/{test_run}_best-model.pt')
        document_embeddings = model.document_embeddings
        print("model loaded")

    # if simple pooled embeddings
    else:
        if test_run == "fasttext":
            document_embeddings = DocumentPoolEmbeddings([WordEmbeddings('en-twitter')])
        elif test_run == "roberta":
            document_embeddings = DocumentPoolEmbeddings(
                [RoBERTaEmbeddings(pretrained_model_name_or_path="roberta-large", layers="21,22,23,24",
                                   pooling_operation="first", use_scalar_mix=True)])
        else:
            print("You need to define proper model name in the code"
                  " or choose from two predefined options: --test_run=fasttext or --test_run=roberta")

    # prepare df for output to csv
    # batch size for embedding tweet instances
    tweets_to_embed = data['text'].copy()
    print("beginning embedding")

    # prepare mini batches
    low_limits = list()
    for x in range(0, len(tweets_to_embed), bs):
        low_limits.append(x)
    up_limits = [x + bs for x in low_limits[:-1]]
    up_limits.append(len(tweets_to_embed))

    # a placeholder for embedded tweets and time of computation
    newEmbedings = list()
    embedding_times = list()

    # embeddings tweets
    for i in range(len(low_limits)):
        it = time.time()
        print(f"batch {math.ceil(up_limits[i] / bs)}")
        # get the list of current tweet instances
        slist = tweets_to_embed.iloc[low_limits[i]:up_limits[i]].to_list()

        # create a list of Sentence objects
        sentlist = list()
        for sent in slist:
            sentlist.append(Sentence(sent, use_tokenizer=True))

        # feed the list of Sentence objects to the model and output embeddings
        document_embeddings.embed(sentlist)

        # add embeddings of sentences to a new data frame
        for num, sentence in enumerate(sentlist):
            sent_emb = sentence.get_embedding()
            newEmbedings.append(sent_emb.squeeze().tolist())

        ft = time.time()
        embedding_times.append((ft - it) / bs)

    print("Average tweet embedding time: ", np.array(embedding_times).mean())
    print("Total tweet embedding time: ", len(tweets_to_embed)*np.array(embedding_times).mean())
    # save all embeddings in a DataFrame
    df = pd.DataFrame(newEmbedings)

    # add rows with target variable and dummy_id for identification of rows
    df = df.astype(np.float16)
    df['sentiment'] = data['sentiment']
    df["dummy_id"] = data["dummy_id"].astype(str)

    print(df.head())

    # if trained embeddings
    if not pool:
        df.to_csv(f"./{dataset}_data/embeddings/{test_run}_encoded_sentences_{fold}{subset}.csv")
    # if pooled embeddings
    else:
        df.to_csv(f"./{dataset}_data/embeddings/{test_run}_encoded_sentences_pooled{subset}.csv")

# if universal sentence encoder (USE)
else:
    # a placeholder for embedded tweets and time of computation
    newEmbedings = list()
    embedding_times = list()

    # import and load USE
    import tensorflow_hub as hub
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    tweets_to_embed = data['text'].to_list()
    it = time.time()
    use_embeddings = embed(tweets_to_embed)

    for _, use_embedding in enumerate(np.array(use_embeddings).tolist()):
        # create a list of embeddings
        newEmbedings.append(use_embedding)

    ft = time.time()
    print("Average tweet embedding time: ", (ft-it)/len(tweets_to_embed))
    print("Total tweet embedding time: ", ft-it)

    # save all embeddings in a DataFrame
    df = pd.DataFrame(newEmbedings)

    # add rows with target variable and dummy_id for identification of rows
    df = df.astype(np.float16)
    df['sentiment'] = data['sentiment']
    df["dummy_id"] = data["dummy_id"].astype(str)

    # output USE embeddings
    df.to_csv(f"./{dataset}_data/embeddings/USE_encoded_sentences{subset}.csv")
    print("USE embeddings saved to: ", f"./{dataset}_data/embeddings/USE_encoded_sentences{subset}.csv")
