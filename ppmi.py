from collections import defaultdict
from tqdm import tqdm

import numpy as np
import pandas as pd


def co_occurrence(sentences, window_size):
    d = defaultdict(int)
    vocab = set()
    for text in tqdm(sentences):
        # preprocessing (use tokenizer instead)
        text = text.lower().split()

        for i in range(len(text)):
            token = text[i]
            vocab.add(token)  # add to vocab
            next_token = text[i+1 : i+1+window_size]
            for t in next_token:
                key = tuple( sorted([t, token]) )
                d[key] += 1

    # formulate the dictionary into dataframe
    vocab = sorted(vocab) # sort vocab
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    return df


def pmi(df, positive=True):
    col_totals = df.sum(axis=0)
    total = col_totals.sum()
    row_totals = df.sum(axis=1)
    expected = np.outer(row_totals, col_totals) / total
    df = df / expected
    with np.errstate(divide='ignore'):
        df = np.log(df)
    df[np.isinf(df)] = 0.0  # log(0) = 0
    if positive:
        df[df < 0] = 0.0
    return df


if __name__ == "__main__":
    text = ["나 는 오늘 밤 에 공부 를 해요 . ", "나 는 오늘 밤 에 공부 를 하지 않아 요 ."]  
    print(pmi(co_occurrence(text, window_size=2), positive=True))
    
    """
           .        공부         나         는         를         밤        않아         에        오늘         요        하지        해요
.   0.000000  0.000000  0.000000  0.000000  0.753772  0.000000  1.446919  0.000000  0.000000  1.734601  0.000000  1.734601
공부  0.000000  0.000000  0.000000  0.000000  0.753772  0.753772  0.000000  0.753772  0.000000  0.000000  0.753772  1.041454
나   0.000000  0.000000  0.000000  1.734601  0.000000  0.000000  0.000000  0.000000  1.446919  0.000000  0.000000  0.000000
는   0.000000  0.000000  1.734601  0.000000  0.000000  1.041454  0.000000  0.000000  1.041454  0.000000  0.000000  0.000000
를   0.753772  0.753772  0.000000  0.000000  0.000000  0.000000  0.753772  0.753772  0.000000  0.000000  0.753772  1.041454
밤   0.000000  0.753772  0.000000  1.041454  0.000000  0.000000  0.000000  0.753772  0.753772  0.000000  0.000000  0.000000
않아  1.446919  0.000000  0.000000  0.000000  0.753772  0.000000  0.000000  0.000000  0.000000  1.734601  1.446919  0.000000
에   0.000000  0.753772  0.000000  0.000000  0.753772  0.753772  0.000000  0.000000  0.753772  0.000000  0.000000  0.000000
오늘  0.000000  0.000000  1.446919  1.041454  0.000000  0.753772  0.000000  0.753772  0.000000  0.000000  0.000000  0.000000
요   1.734601  0.000000  0.000000  0.000000  0.000000  0.000000  1.734601  0.000000  0.000000  0.000000  1.734601  0.000000
하지  0.000000  0.753772  0.000000  0.000000  0.753772  0.000000  1.446919  0.000000  0.000000  1.734601  0.000000  0.000000
해요  1.734601  1.041454  0.000000  0.000000  1.041454  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000  0.000000
    """