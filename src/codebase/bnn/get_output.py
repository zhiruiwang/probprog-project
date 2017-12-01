# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 19:50:14 2017

@author: wang_
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# read training data, test data and groud truth data for test data
train_df = pd.read_csv('../../data/train_FD001.txt', sep=" ", header=None)
test_df = pd.read_csv('../../data/test_FD001.txt', sep=" ", header=None)
truth_df = pd.read_csv('../../data/RUL_FD001.txt', sep=" ", header=None)

# preprocesse the data as described
from util import turbo_preprocessing
train_df, test_df = turbo_preprocessing(train_df,test_df,truth_df)

# pick a window size of 5 cycles
sequence_length = 5


# pick the feature columns 
sensor_cols = ['s' + str(i) for i in range(1,22)]
sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
sequence_cols.extend(sensor_cols)

# generator for the sequences
from util import gen_sequence
seq_gen = [list(gen_sequence(train_df[train_df['id']==id], sequence_length, 
                             sequence_cols)) 
           for id in train_df['id'].unique()]


# generate sequences and convert to numpy array
seq_array = np.concatenate(seq_gen).astype(np.float32)

# generate labels
from util import gen_labels
label_gen = [gen_labels(train_df[train_df['id']==id], sequence_length, ['RUL']) 
             for id in train_df['id'].unique()]
label_array = np.squeeze(np.concatenate(label_gen).astype(np.float32))

seq_array1,seq_array2,label_array1,label_array2 = \
    train_test_split(seq_array,label_array,test_size=0.9,shuffle=False)

H = 100  # number of hidden units
N,_,D = seq_array1.shape  # number of training data points, number of features

#Get output
from models import model_inference_critisism
valid_models = ['Fully Connected Layer','Simple RNN','LSTM','GRU',
          'Two Layer Simple RNN','Two Layer LSTM','Two Layer GRU']
for model in valid_models:
    for bayes in [True, False]:
        model_inference_critisism(model, bayes, seq_array1, 
                                  seq_array2, label_array1, label_array2, 
                                  sequence_length, N, D, H=10, H1=100, H2=50)
