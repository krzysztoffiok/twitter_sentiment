import pandas as pd
import numpy as np
import os

"""
Naive parsing of original SemEval2017 .txt files downloaded from:
http://alt.qcri.org/semeval2017/task4/index.php?id=download-the-full-training-data-for-semeval-2017-task-4
"""


# function to read semeval 2016 data
def txt_to_csv(data_split):
    i = 0
    sentiment = list()
    text = list()

    k = open(f"./semeval_data/source_data/twitter-2016{data_split}-CE.txt", "r")
    for line in k:
        tmp = line.split()
        i += 1
        try:
            if int(tmp[2]) or int(tmp[2]) == 0:
                if len(tmp[2]) > 2:
                    print(tmp)
                sentiment.append(tmp[2])
                text.append(' '.join(tmp[3:]))
        except ValueError:
            try:
                if int(tmp[3]) or int(tmp[3]) == 0:
                    if len(tmp[3]) > 2:
                        print(tmp)
                    sentiment.append(tmp[3])
                    text.append(' '.join(tmp[4:]))
            except ValueError:
                try:

                    if float(tmp[4]) or int(tmp[4]) == 0:
                        if len(tmp[4]) > 2:
                            print(tmp)
                        sentiment.append(tmp[4])
                        text.append(' '.join(tmp[5:]))
                except ValueError:
                    try:
                        if float(tmp[5]) or int(tmp[5]) == 0:
                            sentiment.append(tmp[5])
                            text.append(' '.join(tmp[6:]))
                    except ValueError:

                        if float(tmp[6]) or int(tmp[6]) == 0:
                            sentiment.append(tmp[6])
                            text.append(' '.join(tmp[7:]))

    return pd.DataFrame.from_dict({"sentiment": sentiment, "text": text})


df_semtest = txt_to_csv("test")
df_semtrain = txt_to_csv("train")

df_semtest.to_csv("./semeval_data/source_data/semtest.csv")
df_semtrain.to_csv("./semeval_data/source_data/semtrain.csv")

