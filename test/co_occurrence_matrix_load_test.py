# -*- coding: utf-8 -*-
__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import pickle


with open('../co_occurence_matrix.pkl', 'rb') as f:
    matrix = pickle.load(f)
    f.close()


import math
print(math.log(matrix[4, 4]))
