import numpy as np
import pandas as pd
from sklearn import preprocessing
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen


def load_data(download=True):
    """Load data

    Data Source: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-
    repository/#turbofan

    Reference: Deep Learning Basics for Predictive Maintenance

    Microsoft Azure - LSTMs for Predictive Maintenance
    https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/ \
    Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb

    """
    if download:
        print("Downloading dataset...")
        zipurl = 'https://ti.arc.nasa.gov/c/6/'
        with urlopen(zipurl) as zipresp:
            with ZipFile(BytesIO(zipresp.read())) as zfile:
                zfile.extractall('./CMAPSSData')

    # train set
    train_df = pd.read_csv('CMAPSSData/train_FD001.txt', sep=" ", header=None)
    train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3',
                        's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
                        's10', 's11', 's12', 's13', 's14', 's15', 's16',
                        's17', 's18', 's19', 's20', 's21']
    # test set
    test_df = pd.read_csv('CMAPSSData/test_FD001.txt', sep=" ", header=None)
    test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3',
                       's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9',
                       's10', 's11', 's12', 's13', 's14', 's15', 's16',
                       's17', 's18', 's19', 's20', 's21']
    # ground truth set
    truth_df = pd.read_csv('CMAPSSData/RUL_FD001.txt', sep=" ", header=None)
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
    train_df = train_df.sort_values(['id', 'cycle'])

    return train_df, test_df, truth_df


def preprocess_data(train_df, test_df, truth_df):
    """Preprocess data

    Reference: Deep Learning Basics for Predictive Maintenance

    Microsoft Azure - LSTMs for Predictive Maintenance
    https://github.com/Azure/lstms_for_predictive_maintenance/blob/master/ \
    Deep%20Learning%20Basics%20for%20Predictive%20Maintenance.ipynb

    """
    # train set

    # 1. Data labeling - generate column RUL
    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    train_df = train_df.merge(rul, on=['id'], how='left')
    train_df['RUL'] = train_df['max'] - train_df['cycle']
    train_df.drop('max', axis=1, inplace=True)

    # 3. MinMax normalization
    train_df['cycle_norm'] = train_df['cycle']
    cols_normalize = train_df.columns.difference(
        ['id', 'cycle', 'RUL'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_train_df = pd.DataFrame(
        min_max_scaler.fit_transform(train_df[cols_normalize]),
        columns=cols_normalize, index=train_df.index)
    join_df = train_df[train_df.columns.difference(
        cols_normalize)].join(norm_train_df)
    train_df = join_df.reindex(columns=train_df.columns)

    # test set

    # 1. MinMax normalization
    test_df['cycle_norm'] = test_df['cycle']
    norm_test_df = pd.DataFrame(
        min_max_scaler.transform(test_df[cols_normalize]),
        columns=cols_normalize,
        index=test_df.index)
    test_join_df = test_df[test_df.columns.difference(
        cols_normalize)].join(norm_test_df)
    test_df = test_join_df.reindex(columns=test_df.columns)
    test_df = test_df.reset_index(drop=True)

    # 2. Generate column max for test data
    rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    truth_df.columns = ['more']
    truth_df['id'] = truth_df.index + 1
    truth_df['max'] = rul['max'] + truth_df['more']
    truth_df.drop('more', axis=1, inplace=True)

    # 3. Generate RUL for test data
    test_df = test_df.merge(truth_df, on=['id'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    test_df.drop('max', axis=1, inplace=True)

    return train_df, test_df


def load_and_preprocess(download=True):
    """Load and preprocess dataset
    """
    train_df, test_df, truth_df = load_data(download)
    return preprocess_data(train_df, test_df, truth_df)


def config_data(train_df, seq_length, batch_size):
    """Configure dataset as given condition
    Create fixed-length sequences for given machines
    """
    id_df = train_df['id'].unique()

    def gen_sequence(df, seq_length, col):
        """Generate sequences for given columns and sequence length
        1. No padding.
        2. For testing, drop those which are below the sequence length
        """
        data = df[col].values
        n_record = data.shape[0]
        for s, t in zip(range(0, n_record - seq_length),
                        range(seq_length, n_record)):
            yield data[s:t, :]

    def gen_labels(df, seq_length, label):
        """Generate labels for given columns and sequence length
        Same as generate sequence
        """
        data = df[label].values
        n_record = data.shape[0]
        return data[seq_length:n_record, :]

    col = ['s' + str(i) for i in range(1, 22)]
    col.extend(['setting1', 'setting2', 'setting3', 'cycle_norm'])
    seq_gen = [list(gen_sequence(train_df[train_df['id'] == id],
                                 seq_length, col))
               for id in id_df]

    X_train = np.concatenate(seq_gen).astype(np.float32)

    label_gen = [gen_labels(train_df[train_df['id'] == id],
                            seq_length, ['RUL'])
                 for id in id_df]

    y_train = np.squeeze(np.concatenate(label_gen).astype(np.float32))

    return X_train, y_train
