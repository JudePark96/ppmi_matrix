# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

corpus_path = '../corpus/processed_kp20k_training_context_filtered_RmKeysAllUnk.txt'
corpus = []

with open(corpus_path, 'r', encoding='utf-8') as f:
    for idx, line in enumerate(f):
        if idx % 1000 == 0:
            logger.info(f'{idx} -> {line}')
        corpus.append(line.strip().split(' '))
    f.close()
