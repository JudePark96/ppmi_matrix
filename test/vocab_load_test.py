# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import pickle


vocab2idx_path = '../vocab/kp20k_vocab2idx.pkl'
idx2vocab_path = '../vocab/kp20k_idx2vocab.pkl'

with open(vocab2idx_path, 'rb') as f:
    vocab2idx = pickle.load(f)
    f.close()

with open(idx2vocab_path, 'rb') as f:
    idx2vocab = pickle.load(f)
    f.close()

assert len(vocab2idx) == len(idx2vocab)
assert type(vocab2idx) == dict
assert type(idx2vocab) == list