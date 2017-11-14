import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing

"""
Load and preprocess datasets

Reference: Deep Learning Basics for Predictive Maintenance

Microsoft Azure - LSTMs for Predictive Maintenance
https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/ \
    Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb
"""

# Download data


# Load data

## train set
train_df = pd.read_csv('CMAPSSData/train_FD001.txt', sep=" ", header=None)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2',
                    's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12',
                    's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
## test set
test_df = pd.read_csv('CMAPSSData/test_FD001.txt', sep=" ", header=None)
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2',
                   's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12',
                   's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
## ground truth set
truth_df = pd.read_csv('CMAPSSData/RUL_FD001.txt', sep=" ", header=None)
truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
train_df = train_df.sort_values(['id', 'cycle'])


# Preprocess data

# train set

## 1. Data labeling - generate column RUL
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_df = train_df.merge(rul, on=['id'], how='left')
train_df['RUL'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)

## 2. Generate label columns for training data
w1 = 30
w0 = 15
train_df['label1'] = np.where(train_df['RUL'] <= w1, 1, 0)
train_df['label2'] = train_df['label1']
train_df.loc[train_df['RUL'] <= w0, 'label2'] = 2

## 3. MinMax normalization
train_df['cycle_norm'] = train_df['cycle']
cols_normalize = train_df.columns.difference(
    ['id', 'cycle', 'RUL', 'label1', 'label2'])
min_max_scaler = MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                             columns=cols_normalize, index=train_df.index)
join_df = train_df[train_df.columns.difference(
    cols_normalize)].join(norm_train_df)
train_df = join_df.reindex(columns=train_df.columns)


# test set
## 1. MinMax normalization
test_df['cycle_norm'] = test_df['cycle']
norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]),
                            columns=cols_normalize,
                            index=test_df.index)
test_join_df = test_df[test_df.columns.difference(
    cols_normalize)].join(norm_test_df)
test_df = test_join_df.reindex(columns=test_df.columns)
test_df = test_df.reset_index(drop=True)

## 2. Generate column max for test data
rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
truth_df.columns = ['more']
truth_df['id'] = truth_df.index + 1
truth_df['max'] = rul['max'] + truth_df['more']
truth_df.drop('more', axis=1, inplace=True)

## 3. Generate RUL for test data
test_df = test_df.merge(truth_df, on=['id'], how='left')
test_df['RUL'] = test_df['max'] - test_df['cycle']
test_df.drop('max', axis=1, inplace=True)

## 4. Generate label columns w0 and w1 for test data
test_df['label1'] = np.where(test_df['RUL'] <= w1, 1, 0)
test_df['label2'] = test_df['label1']
test_df.loc[test_df['RUL'] <= w0, 'label2'] = 2
