# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import argparse
import pickle
import logging
from collections import defaultdict

from coocurrence import build_co_occur_matrix
from tqdm import tqdm


def main():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--v2i_path', type=str, required=True, help='vocab2idx file path')
    parser.add_argument('--i2v_path', type=str, required=True, help='idx2vocab file path')
    parser.add_argument('--training_corpus', type=str, required=True, help='training corpus file path')
    parser.add_argument('--output_path', type=str, required=True, help='output file path')
    parser.add_argument('--windows', type=int, default=3, help='value of sliding window')

    args = parser.parse_args()

    corpus = []

    with open(args.training_corpus, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            corpus.append(line.strip().split(' '))
        f.close()

    with open(args.v2i_path, 'rb') as f:
        vocab2idx = defaultdict(int, pickle.load(f))
        f.close()

    x = build_co_occur_matrix(corpus, vocab2idx, args.windows, dynamic_weight=True, verbose=True)

    with open(args.output_path, 'wb') as f:
        pickle.dump(x, f)
        f.close()

    logger.info('saving co-occurrence matrix done.')


if __name__ == '__main__':
    main()