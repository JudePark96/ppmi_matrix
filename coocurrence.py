# -*- coding: utf-8 -*-
# this code revised from https://github.com/lovit/soynlp/blob/3497e958035e24f9204d91572252fae61791e6d7/soynlp/vectorizer/_word_context.py
# I edited some of above code.

__email__ = 'judepark@kookmin.ac.kr'

import logging

from scipy.sparse import csr_matrix
from collections import defaultdict
from typing import List

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def build_word_context(
        sents: List[str],
        windows: int,
        dynamic_weight: bool,
        vocab2idx: dict,
        verbose=True):
    # TODO: Sents 는 무조건 sent.strip().split(' ') 한 것들의 리스트를 받을 것.
    word2contexts = defaultdict(lambda: defaultdict(int))

    if dynamic_weight:
        weight = [(windows - i) / windows for i in range(windows)]
    else:
        weight = [1] * windows

    for i_sent, sent in enumerate(sents):
        if verbose and i_sent % 1000 == 0:
            logger.info(f'{i_sent} - scanning (word, context) pairs')

        # sents (List of sentences) has already tokenized sentences.
        n = len(sent)

        for i, word in enumerate(sents):
            if not (word in vocab2idx):
                continue

            for w in range(windows):
                j = i - (w + 1)
                if j < 0 or not (sent[j] in vocab2idx):
                    continue
                word2contexts[word][sent[j]] += weight[w]

            for w in range(windows):
                j = i + w + 1
                if j >= n or not (sent[j] in vocab2idx):
                    continue
                word2contexts[word][sent[j]] += weight[w]

    return word2contexts

def encode_as_matrix(word2contexts: dict, vocab2idx: dict, verbose: bool):
    rows = []
    cols = []
    data = []
    for word, contexts in word2contexts.items():
        word_idx = vocab2idx[word]
        for context, cooccurrence in contexts.items():
            context_idx = vocab2idx[context]
            rows.append(word_idx)
            cols.append(context_idx)
            data.append(cooccurrence)
            print(word, context, cooccurrence)
    x = csr_matrix((data, (rows, cols)))

    if verbose:
        logger.info(f'(word, context) matrix was constructed. shape = {x.shape}')

    return x

def build_co_occur_matrix(
        sents: List[str],
        vocab2idx: dict,
        windows: int = 3,
        dynamic_weight: bool = True,
        verbose: bool = True):
    if verbose:
        logger.info('create (word, contexts) matrix')

    word2contexts = build_word_context(
        sents, windows, dynamic_weight, vocab2idx, verbose)

    x = encode_as_matrix(word2contexts, vocab2idx, verbose)

    if verbose:
        logger.info('building co-occurrence matrix done.')
    return x
