import pandas as pd
import argparse
import numpy as np

""" example use to create tweet-level embeddings

# with LSTM training over the pre-trained model
python3 embed_sentences_flair.py --fold=5 --test_run=fasttext
python3 embed_sentences_flair.py --fold=5 --test_run=roberta_lstm
Important: needs to be run separately for each fold e.g. with bash command:
for i in {0..4}; do python3 embed_sentences_flair.py --fold=$i --test_run=fasttext; done

# no LSTM, just mean of pre-trained token embeddings without fine-tuning
python3 embed_sentences_flair.py --pool=True  --test_run=fasttext
python3 embed_sentences_flair.py --pool=True  --test_run=roberta

# CLS token output of fine-tuned transformer model
python3 embed_sentences_flair.py --fold=0 --test_run=roberta_large_ft
Important: needs to be run separately for each fold e.g. with bash command:
for i in {0..4}; do python3 embed_sentences_flair.py --fold=$i --test_run=roberta_large_ft; done

# universal sentence encoder
python3 embed_sentences_flair.py --use=True

"""

parser = argparse.ArgumentParser(description='Classify data')
parser.add_argument('--test_run', required=False, default='',
                    type=str, help='name of model')
parser.add_argument("--nrows", required=False, default=6000, type=int)
parser.add_argument("--fold", required=False, default=5, type=int)
parser.add_argument("--file", required=False, default="tweet_sentiment_input_file", type=str)
parser.add_argument("--pool", required=False, default=False, type=bool)
parser.add_argument("--use", required=False, default=False, type=bool)
args = parser.parse_args()

fold = args.fold
test_run = args.test_run
nrows = args.nrows
file_name = args.file
pool = args.pool
_use = args.use

# read data
df = pd.read_excel(f"./data/source_data/{file_name}.xlsx", converters={'dummy_id': str})
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
    flair.device = torch.device('cuda')

    # load various embeddings
    from flair.embeddings import WordEmbeddings, DocumentPoolEmbeddings, RoBERTaEmbeddings
    from flair.data import Sentence

    # if trained embeddings
    if not pool:
        # embeddings trained to the "downstream task"
        model = TextClassifier.load(f'./data/model_sentiment_{fold}/{test_run}_best-model.pt')
        document_embeddings = model.document_embeddings

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

    # prepare df for output csv
    hidden_size = document_embeddings.embedding_length
    columns = [x for x in range(hidden_size)]
    df = pd.DataFrame(columns=[columns])

    for num, sent in enumerate(data['text']):
        # prepare each tweet for flair tokenized format
        sentence = Sentence(sent, use_tokenizer=True)
        # embed sentence
        document_embeddings.embed(sentence)
        sent_emb = sentence.get_embedding()
        # add new row to df
        df.loc[num] = sent_emb.squeeze().tolist()

    # add rows with target variable and dummy_id for identification of rows
    df = df.astype(np.float16)
    df['sentiment'] = data['sentiment']
    df["dummy_id"] = data["dummy_id"].astype(str)

    print(df.head())

    # if LSTM trained embeddings
    if not pool:
        df.to_csv(f"./data/embeddings/{test_run}_encoded_sentences_{fold}.csv")
    # if pooled embeddings
    else:
        df.to_csv(f"./data/embeddings/{test_run}_encoded_sentences_pooled.csv")

# if universal sentence encoder (USE)
else:
    # prepare df for output csv
    hidden_size = 512
    columns = [x for x in range(hidden_size)]
    df = pd.DataFrame(columns=[columns])

    # import and load USE
    import tensorflow_hub as hub
    import numpy as np
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
    sentences_to_embed = data['text'].to_list()
    use_embeddings = embed(sentences_to_embed)

    for i, use_embedding in enumerate(np.array(use_embeddings).tolist()):
        # add new row to df
        df.loc[i] = use_embedding

    # add rows with target variable and dummy_id for identification of rows
    df = df.astype(np.float16)
    df['sentiment'] = data['sentiment']
    df["dummy_id"] = data["dummy_id"].astype(str)

    # output USE embeddings
    df.to_csv("./data/embeddings/USE_encoded_sentences.csv")
    print("USE embeddings saved to: ", "./data/embeddings/USE_encoded_sentences.csv")
