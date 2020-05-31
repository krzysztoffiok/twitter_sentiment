import pandas as pd
import numpy as np
import datatable as dt
import re
"""
Basic pre-processing of Twitter text from SemEval2017 data set.
"""


# replace repeating characters so that only 2 repeats remain
def repoo(x):
    repeat_regexp = re.compile(r'(\S+)(\1{2,})')
    repl = r'\2'
    return repeat_regexp.sub(repl=r'\2', string=x)


file_names = ["./semeval_data/source_data/semtrain.csv", "./semeval_data/source_data/semtest.csv"]
for file_name in file_names:
    df = dt.fread(file_name).to_pandas()

    df_sampled = df.copy()
    sample_size = len(df_sampled)
    # preprocess data
    import re

    # change all pic.twitter.com to "IMAGE"
    df_sampled["text"] = df_sampled["text"].str.replace(
        'pic.twitter.com/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' _IMAGE ', regex=True)

    # # get rid of some instances of IMG
    df_sampled["text"] = df_sampled["text"].str.replace(
        'https://pbs.twimg.com/media/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 'IMAGE ',
        regex=True)

    # get rid of some instances of https://twitter.com -> to RETWEET
    df_sampled["text"] = df_sampled["text"].str.replace(
        'https://twitter.com(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' _RETWEET ',
        regex=True)

    # change all URLS to "URL"

    df_sampled["text"] = df_sampled["text"].str.replace(
        'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' _URL ', regex=True)

    # get rid of character repeats
    for i in range(10):
        df_sampled["text"] = df_sampled["text"].map(lambda x: repoo(str(x)))

    # get rid of endline signs
    df_sampled["text"] = df_sampled["text"].str.replace("\n", "")

    # save to file the sampled DF
    df_sampled[["sentiment", "text"]].to_csv(f"{file_name[:-4]}_filtered.csv")
