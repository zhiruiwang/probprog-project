import numpy as np
import pandas as pd
import tensorflow as tf
import edward as ed
from edward.models import Normal
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras.api.keras.layers import SimpleRNN
from tensorflow.contrib.keras.api.keras.layers import LSTM
from tensorflow.contrib.keras.api.keras.layers import GRU
from tensorflow.contrib.keras.api.keras.optimizers import Nadam


plt.style.use('seaborn-talk')
sns.set_context("talk", font_scale=1.4)
sess = ed.get_session()


def neural_network_with_2_layers(x, W_0, W_1, b_0, b_1):
    h = tf.nn.tanh(tf.matmul(x, W_0) + b_0)
    h = tf.matmul(h, W_1) + b_1
    return tf.reshape(h, [-1])


def rnn_cell_sig(hprev, xt, Wh, Wx, bh):
    return tf.sigmoid(tf.matmul(hprev, Wh) + tf.matmul(xt, Wx) + bh)


def rnn_cell_tanh(hprev, xt, Wh, Wx, bh):
    return tf.tanh(tf.matmul(hprev, Wh) + tf.matmul(xt, Wx) + bh)


def rnn_layer(X, Wh, Wx, bh, Wy, by, H):
    N, sequence_length, D = X.get_shape().as_list()
    h = tf.zeros([N, H])
    for i in range(sequence_length):
        h = rnn_cell_tanh(h, tf.squeeze(
            tf.slice(X, [0, i, 0], [N, 1, D])), Wh, Wx, bh)
    return tf.reshape(tf.matmul(h, Wy) + by, [-1])


def LSTM_cell(hprev, cprev, xt, Wf, Uf, Wi, Ui, Wo, Uo, Wc, Uc,
              bf, bi, bo, bc):
    f = rnn_cell_sig(hprev, xt, Wf, Uf, bf)
    i = rnn_cell_sig(hprev, xt, Wi, Ui, bi)
    o = rnn_cell_sig(hprev, xt, Wo, Uo, bo)
    c = f*cprev + i*rnn_cell_tanh(hprev, xt, Wc, Uc, bc)
    h = o*tf.tanh(c)
    return h, c


def LSTM_layer(X, Wf, Uf, Wi, Ui, Wo, Uo, Wc, Uc,
               bf, bi, bo, bc, Wy, by, H):
    N, sequence_length, D = X.get_shape().as_list()
    h = tf.zeros([N, H])
    c = tf.zeros([N, H])
    for i in range(sequence_length):
        h, c = LSTM_cell(h, c, tf.squeeze(tf.slice(
            X, [0, i, 0], [N, 1, D])), Wf, Uf, Wi, Ui, Wo, Uo,
            Wc, Uc, bf, bi, bo, bc)
    return tf.reshape(tf.matmul(h, Wy) + by, [-1])


def GRU_cell(hprev, xt, Wz, Uz, Wr, Ur, Wh, Uh, bz, br, bh):
    z = rnn_cell_sig(hprev, xt, Wz, Uz, bz)
    r = rnn_cell_sig(hprev, xt, Wr, Ur, br)
    h = z*hprev + (1-z)*rnn_cell_tanh(r*hprev, xt, Wh, Uh, bh)
    return h


def GRU_layer(X, Wz, Uz, Wr, Ur, Wh, Uh, bz, br, bh, Wy, by, H):
    N, sequence_length, D = X.get_shape().as_list()
    h = tf.zeros([N, H])
    for i in range(sequence_length):
        h = GRU_cell(h, tf.squeeze(
            tf.slice(X, [0, i, 0], [N, 1, D])), Wz, Uz, Wr, Ur,
            Wh, Uh, bz, br, bh)
    return tf.reshape(tf.matmul(h, Wy) + by, [-1])


def two_rnn_layer(X, Wh1, Wx1, bh1, Wh2, Wx2, bh2, Wy, by, H1, H2):
    N, sequence_length, D = X.get_shape().as_list()
    h1 = tf.zeros([N, H1])
    h2 = tf.zeros([N, H2])
    for i in range(sequence_length):
        h1 = rnn_cell_tanh(h1, tf.squeeze(
            tf.slice(X, [0, i, 0], [N, 1, D])), Wh1, Wx1, bh1)
        h2 = rnn_cell_tanh(h2, h1, Wh2, Wx2, bh2)
    return tf.reshape(tf.matmul(h2, Wy) + by, [-1])


def two_LSTM_layer(X, Wf1, Uf1, Wi1, Ui1, Wo1, Uo1, Wc1, Uc1, bf1,
                   bi1, bo1, bc1, Wf2, Uf2, Wi2, Ui2,
                   Wo2, Uo2, Wc2, Uc2, bf2, bi2, bo2,
                   bc2, Wy, by, H1, H2):
    N, sequence_length, D = X.get_shape().as_list()
    h1 = tf.zeros([N, H1])
    c1 = tf.zeros([N, H1])
    h2 = tf.zeros([N, H2])
    c2 = tf.zeros([N, H2])
    for i in range(sequence_length):
        h1, c1 = LSTM_cell(h1, c1, tf.squeeze(
            tf.slice(X, [0, i, 0], [N, 1, D])), Wf1, Uf1, Wi1, Ui1,
            Wo1, Uo1, Wc1, Uc1, bf1, bi1, bo1, bc1)
        h2, c2 = LSTM_cell(h2, c2, h1, Wf2, Uf2, Wi2, Ui2, Wo2,
                           Uo2, Wc2, Uc2, bf2, bi2, bo2, bc2)

    return tf.reshape(tf.matmul(h2, Wy) + by, [-1])


def two_GRU_layer(X, Wz1, Uz1, Wr1, Ur1, Wh1, Uh1, bz1, br1, bh1,
                  Wz2, Uz2, Wr2, Ur2, Wh2, Uh2, bz2, br2, bh2,
                  Wy, by, H1, H2):
    N, sequence_length, D = X.get_shape().as_list()
    h1 = tf.zeros([N, H1])
    h2 = tf.zeros([N, H2])
    for i in range(sequence_length):
        h1 = GRU_cell(h1, tf.squeeze(
            tf.slice(X, [0, i, 0], [N, 1, D])), Wz1, Uz1,
            Wr1, Ur1, Wh1, Uh1, bz1, br1, bh1)
        h2 = GRU_cell(h2, h1, Wz2, Uz2, Wr2, Ur2, Wh2, Uh2, bz2, br2, bh2)
    return tf.reshape(tf.matmul(h2, Wy) + by, [-1])


def model_inference_critisism(model_name, Bayesian, seq_array1, seq_array2,
                              label_array1, label_array2, sequence_length,
                              N, D, H=100, H1=100, H2=50):
    less_50 = label_array2 <=50
    if not Bayesian:
        model = Sequential()
        if model_name == 'Fully Connected Layer':
            model.add(Dense(H,
                            input_shape=[D], activation='tanh'))
        elif model_name == 'Simple RNN':
            model.add(SimpleRNN(
                input_shape=(sequence_length, D),
                units=H))
        elif model_name == 'LSTM':
            model.add(LSTM(
                input_shape=(sequence_length, D),
                units=H))
        elif model_name == 'GRU':
            model.add(GRU(
                input_shape=(sequence_length, D),
                units=H))
            model.add(Dense(units=1))
        elif model_name == 'Two Layer Simple RNN':
            model.add(SimpleRNN(
                input_shape=(sequence_length, D),
                units=H1, return_sequences=True))
            model.add(SimpleRNN(
                units=H2))
        elif model_name == 'Two Layer LSTM':
            model.add(LSTM(
                input_shape=(sequence_length, D),
                units=H1, return_sequences=True))
            model.add(LSTM(
                units=H2))
        elif model_name == 'Two Layer GRU':
            model.add(GRU(
                input_shape=(sequence_length, D),
                units=H1, return_sequences=True))
            model.add(GRU(
                units=H2))
        else:
            raise Exception('Please specify a valid model!')
        model.add(Dense(units=1))
        if model_name == 'Fully Connected Layer':
            # inference
            nadam = Nadam(lr=0.05)
            model.compile(loss='mean_squared_error',
                          optimizer=nadam, metrics=['mean_squared_error'])
            model.fit(seq_array1[:, sequence_length-1, :], label_array1,
                      epochs=300, batch_size=200,
                      validation_data=(seq_array2[:, sequence_length-1, :],
                                       label_array2),
                      verbose=0)
            y_pred = np.squeeze(model.predict(
                seq_array2[:, sequence_length-1, :]))
        else:
            model.compile(loss='mean_squared_error',
                          optimizer='nadam', metrics=['mean_squared_error'])
            model.fit(seq_array1, label_array1, epochs=300, batch_size=200,
                      validation_data=(seq_array2, label_array2), verbose=0)
            y_pred = np.squeeze(model.predict(seq_array2))

        # Critisim

        # Histogram
        sns.distplot(y_pred)
        plt.title('Histogram of RUL, Frequentist {}'.format(model_name))
        plt.xlabel('RUL')
        plt.show()

        # RMSE
        print('Validation RMSE: {}'.format(np.sqrt(
            mean_squared_error(label_array2, y_pred))))
        print('Validation RMSE for RUL under 50: {}'.format(np.sqrt(
            mean_squared_error(label_array2[less_50], y_pred[less_50]))))

        # Prediction time series
        pd.DataFrame([label_array2, y_pred]).transpose().rename(
            columns={0: 'True', 1: 'Pred'})[-1500:].plot()
        plt.title('Prediction of RUL, Frequentist {}'.format(model_name))
        plt.xlabel('RUL')
        plt.show()

    elif Bayesian:
        if model_name == 'Fully Connected Layer':
            W_0 = Normal(loc=tf.zeros([D, H]),
                         scale=tf.ones([D, H]))
            W_1 = Normal(loc=tf.zeros([H, 1]),
                         scale=tf.ones([H, 1]))
            b_0 = Normal(loc=tf.zeros(H),
                         scale=tf.ones(H))
            b_1 = Normal(loc=tf.zeros(1),
                         scale=tf.ones(1))

            x = tf.placeholder(tf.float32, [N, D])
            y = Normal(loc=neural_network_with_2_layers(x, W_0, W_1, b_0, b_1),
                       scale=tf.ones(N) * 0.1)  # constant noise
            # BACKWARD MODEL A
            q_W_0 = Normal(loc=tf.Variable(tf.random_normal([D, H])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H]))))
            q_W_1 = Normal(loc=tf.Variable(tf.random_normal([H, 1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H, 1]))))
            q_b_0 = Normal(loc=tf.Variable(tf.random_normal([H])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H]))))
            q_b_1 = Normal(loc=tf.Variable(tf.random_normal([1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([1]))))
            # INFERENCE A
            # this will take a couple of minutes
            inference = ed.KLqp(latent_vars={W_0: q_W_0, b_0: q_b_0,
                                             W_1: q_W_1, b_1: q_b_1},
                                data={x: seq_array1[:, sequence_length-1, :],
                                      y: label_array1})
            inference.run(n_samples=5, n_iter=25000)
            xp = tf.placeholder(tf.float32, seq_array2[
                                :, sequence_length-1, :].shape)
            y_preds = [sess.run(neural_network_with_2_layers(xp, q_W_0, q_W_1,
                                                             q_b_0, q_b_1),
                                {xp: seq_array2[:, sequence_length-1, :]})
                       for _ in range(50)]
        elif model_name == 'Simple RNN':
            Wh = Normal(loc=tf.zeros([H, H]), scale=tf.ones([H, H]))
            Wx = Normal(loc=tf.zeros([D, H]), scale=tf.ones([D, H]))
            Wy = Normal(loc=tf.zeros([H, 1]), scale=tf.ones([H, 1]))
            bh = Normal(loc=tf.zeros(H), scale=tf.ones(H))
            by = Normal(loc=tf.zeros(1), scale=tf.ones(1))

            X = tf.placeholder(tf.float32, [N, sequence_length, D])
            # X = tf.placeholder(tf.float32,[sequence_length,N,D])
            y = Normal(loc=rnn_layer(X, Wh, Wx, bh, Wy, by, H), scale=1.)
            # BACKWARD MODEL A
            q_Wh = Normal(loc=tf.Variable(tf.random_normal([H, H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H, H]))))
            q_Wx = Normal(loc=tf.Variable(tf.random_normal([D, H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H]))))
            q_Wy = Normal(loc=tf.Variable(tf.random_normal([H, 1])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H, 1]))))
            q_bh = Normal(loc=tf.Variable(tf.random_normal([H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H]))))
            q_by = Normal(loc=tf.Variable(tf.random_normal([1])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([1]))))
            # INFERENCE A
            # this will take a couple of minutes
            inference = ed.KLqp(latent_vars={Wh: q_Wh, bh: q_bh,
                                             Wx: q_Wx, Wy: q_Wy, by: q_by},
                                data={X: seq_array1, y: label_array1})
            inference.run(n_samples=5, n_iter=2500)
            Xp = tf.placeholder(tf.float32, seq_array2.shape)
            y_preds = [sess.run(rnn_layer(Xp, q_Wh, q_Wx, q_bh, q_Wy, q_by, H),
                                {Xp: seq_array2})
                       for _ in range(50)]
        elif model_name == 'LSTM':
            Wf = Normal(loc=tf.zeros([H, H]), scale=tf.ones([H, H]))
            Uf = Normal(loc=tf.zeros([D, H]), scale=tf.ones([D, H]))
            Wi = Normal(loc=tf.zeros([H, H]), scale=tf.ones([H, H]))
            Ui = Normal(loc=tf.zeros([D, H]), scale=tf.ones([D, H]))
            Wo = Normal(loc=tf.zeros([H, H]), scale=tf.ones([H, H]))
            Uo = Normal(loc=tf.zeros([D, H]), scale=tf.ones([D, H]))
            Wc = Normal(loc=tf.zeros([H, H]), scale=tf.ones([H, H]))
            Uc = Normal(loc=tf.zeros([D, H]), scale=tf.ones([D, H]))
            Wy = Normal(loc=tf.zeros([H, 1]), scale=tf.ones([H, 1]))
            bf = Normal(loc=tf.zeros(H), scale=tf.ones(H))
            bi = Normal(loc=tf.zeros(H), scale=tf.ones(H))
            bo = Normal(loc=tf.zeros(H), scale=tf.ones(H))
            bc = Normal(loc=tf.zeros(H), scale=tf.ones(H))
            by = Normal(loc=tf.zeros(1), scale=tf.ones(1))

            X = tf.placeholder(tf.float32, [N, sequence_length, D])
            y = Normal(loc=LSTM_layer(X, Wf, Uf, Wi, Ui, Wo, Uo,
                                      Wc, Uc, bf, bi, bo, bc, Wy, by, H),
                       scale=1.)
            # BACKWARD MODEL A
            q_Wf = Normal(loc=tf.Variable(tf.random_normal([H, H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H, H]))))
            q_Uf = Normal(loc=tf.Variable(tf.random_normal([D, H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H]))))
            q_Wi = Normal(loc=tf.Variable(tf.random_normal([H, H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H, H]))))
            q_Ui = Normal(loc=tf.Variable(tf.random_normal([D, H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H]))))
            q_Wo = Normal(loc=tf.Variable(tf.random_normal([H, H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H, H]))))
            q_Uo = Normal(loc=tf.Variable(tf.random_normal([D, H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H]))))
            q_Wc = Normal(loc=tf.Variable(tf.random_normal([H, H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H, H]))))
            q_Uc = Normal(loc=tf.Variable(tf.random_normal([D, H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H]))))
            q_Wy = Normal(loc=tf.Variable(tf.random_normal([H, 1])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H, 1]))))
            q_bf = Normal(loc=tf.Variable(tf.random_normal([H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H]))))
            q_bi = Normal(loc=tf.Variable(tf.random_normal([H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H]))))
            q_bo = Normal(loc=tf.Variable(tf.random_normal([H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H]))))
            q_bc = Normal(loc=tf.Variable(tf.random_normal([H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H]))))
            q_by = Normal(loc=tf.Variable(tf.random_normal([1])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([1]))))
            # INFERENCE A
            # this will take a couple of minutes
            inference = ed.KLqp(latent_vars={Wf: q_Wf, Uf: q_Uf, Wi: q_Wi,
                                             Ui: q_Ui, Wo: q_Wo, Uo: q_Uo,
                                             Wc: q_Wc, Uc: q_Uc, bf: q_bf,
                                             bi: q_bi, bo: q_bo, bc: q_bc,
                                             Wy: q_Wy, by: q_by},
                                data={X: seq_array1, y: label_array1})
            inference.run(n_samples=5, n_iter=2500)
            Xp = tf.placeholder(tf.float32, seq_array2.shape)
            y_preds = [sess.run(LSTM_layer(Xp, q_Wf, q_Uf, q_Wi, q_Ui, q_Wo,
                                           q_Uo, q_Wc, q_Uc, q_bf, q_bi,
                                           q_bo, q_bc, q_Wy, q_by, H),
                                {Xp: seq_array2}) for _ in range(50)]

        elif model_name == 'GRU':
            Wz = Normal(loc=tf.zeros([H, H]), scale=tf.ones([H, H]))
            Uz = Normal(loc=tf.zeros([D, H]), scale=tf.ones([D, H]))
            Wr = Normal(loc=tf.zeros([H, H]), scale=tf.ones([H, H]))
            Ur = Normal(loc=tf.zeros([D, H]), scale=tf.ones([D, H]))
            Wh = Normal(loc=tf.zeros([H, H]), scale=tf.ones([H, H]))
            Uh = Normal(loc=tf.zeros([D, H]), scale=tf.ones([D, H]))
            Wy = Normal(loc=tf.zeros([H, 1]), scale=tf.ones([H, 1]))
            bz = Normal(loc=tf.zeros(H), scale=tf.ones(H))
            br = Normal(loc=tf.zeros(H), scale=tf.ones(H))
            bh = Normal(loc=tf.zeros(H), scale=tf.ones(H))
            by = Normal(loc=tf.zeros(1), scale=tf.ones(1))

            X = tf.placeholder(tf.float32, [N, sequence_length, D])
            y = Normal(loc=GRU_layer(X, Wz, Uz, Wr, Ur, Wh,
                                     Uh, bz, br, bh, Wy, by, H), scale=1.)

            # BACKWARD MODEL A
            q_Wz = Normal(loc=tf.Variable(tf.random_normal([H, H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H, H]))))
            q_Uz = Normal(loc=tf.Variable(tf.random_normal([D, H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H]))))
            q_Wr = Normal(loc=tf.Variable(tf.random_normal([H, H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H, H]))))
            q_Ur = Normal(loc=tf.Variable(tf.random_normal([D, H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H]))))
            q_Wh = Normal(loc=tf.Variable(tf.random_normal([H, H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H, H]))))
            q_Uh = Normal(loc=tf.Variable(tf.random_normal([D, H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H]))))
            q_Wy = Normal(loc=tf.Variable(tf.random_normal([H, 1])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H, 1]))))
            q_bz = Normal(loc=tf.Variable(tf.random_normal([H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H]))))
            q_br = Normal(loc=tf.Variable(tf.random_normal([H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H]))))
            q_bh = Normal(loc=tf.Variable(tf.random_normal([H])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H]))))
            q_by = Normal(loc=tf.Variable(tf.random_normal([1])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([1]))))
            # INFERENCE A
            # this will take a couple of minutes
            inference = ed.KLqp(latent_vars={Wz: q_Wz, Uz: q_Uz, Wr: q_Wr,
                                             Ur: q_Ur, Wh: q_Wh, Uh: q_Uh,
                                             bz: q_bz, bz: q_bz, bh: q_bh,
                                             Wy: q_Wy, by: q_by},
                                data={X: seq_array1,
                                      y: label_array1})
            inference.run(n_samples=5, n_iter=2500)
            Xp = tf.placeholder(tf.float32, seq_array2.shape)
            y_preds = [sess.run(GRU_layer(Xp, q_Wz, q_Uz, q_Wr, q_Ur, q_Wh,
                                          q_Uh, q_bz, q_br, q_bh,
                                          q_Wy, q_by, H),
                                {Xp: seq_array2})
                       for _ in range(50)]

        elif model_name == 'Two Layer Simple RNN':
            Wh1 = Normal(loc=tf.zeros([H1, H1]), scale=tf.ones([H1, H1]))
            Wx1 = Normal(loc=tf.zeros([D, H1]), scale=tf.ones([D, H1]))
            Wh2 = Normal(loc=tf.zeros([H2, H2]), scale=tf.ones([H2, H2]))
            Wx2 = Normal(loc=tf.zeros([H1, H2]), scale=tf.ones([H1, H2]))
            Wy = Normal(loc=tf.zeros([H2, 1]), scale=tf.ones([H2, 1]))
            bh1 = Normal(loc=tf.zeros(H1), scale=tf.ones(H1))
            bh2 = Normal(loc=tf.zeros(H2), scale=tf.ones(H2))
            by = Normal(loc=tf.zeros(1), scale=tf.ones(1))

            X = tf.placeholder(tf.float32, [N, sequence_length, D])
            y = Normal(loc=two_rnn_layer(X, Wh1, Wx1, bh1, Wh2,
                                         Wx2, bh2, Wy, by, H1, H2), scale=1.)
            # BACKWARD MODEL A
            q_Wh1 = Normal(loc=tf.Variable(tf.random_normal([H1, H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1, H1]))))
            q_Wx1 = Normal(loc=tf.Variable(tf.random_normal([D, H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H1]))))
            q_Wh2 = Normal(loc=tf.Variable(tf.random_normal([H2, H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2, H2]))))
            q_Wx2 = Normal(loc=tf.Variable(tf.random_normal([H1, H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1, H2]))))
            q_Wy = Normal(loc=tf.Variable(tf.random_normal([H2, 1])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2, 1]))))
            q_bh1 = Normal(loc=tf.Variable(tf.random_normal([H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1]))))
            q_bh2 = Normal(loc=tf.Variable(tf.random_normal([H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2]))))
            q_by = Normal(loc=tf.Variable(tf.random_normal([1])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([1]))))

            # INFERENCE A
            # this will take a couple of minutes
            inference = ed.KLqp(latent_vars={Wh1: q_Wh1, bh1: q_bh1,
                                             Wh2: q_Wh2, bh2: q_bh2,
                                             Wx1: q_Wx1, Wx2: q_Wx2, Wy: q_Wy,
                                             by: q_by},
                                data={X: seq_array1,
                                      y: label_array1})
            inference.run(n_samples=5, n_iter=2500)
            Xp = tf.placeholder(tf.float32, seq_array2.shape)
            y_preds = [sess.run(two_rnn_layer(Xp, q_Wh1, q_Wx1, q_bh1,
                                              q_Wh2, q_Wx2, q_bh2, q_Wy, q_by,
                                              H1, H2), {Xp: seq_array2})
                       for _ in range(50)]
        elif model_name == 'Two Layer LSTM':
            Wf1 = Normal(loc=tf.zeros([H1, H1]), scale=tf.ones([H1, H1]))
            Uf1 = Normal(loc=tf.zeros([D, H1]), scale=tf.ones([D, H1]))
            Wi1 = Normal(loc=tf.zeros([H1, H1]), scale=tf.ones([H1, H1]))
            Ui1 = Normal(loc=tf.zeros([D, H1]), scale=tf.ones([D, H1]))
            Wo1 = Normal(loc=tf.zeros([H1, H1]), scale=tf.ones([H1, H1]))
            Uo1 = Normal(loc=tf.zeros([D, H1]), scale=tf.ones([D, H1]))
            Wc1 = Normal(loc=tf.zeros([H1, H1]), scale=tf.ones([H1, H1]))
            Uc1 = Normal(loc=tf.zeros([D, H1]), scale=tf.ones([D, H1]))

            Wf2 = Normal(loc=tf.zeros([H2, H2]), scale=tf.ones([H2, H2]))
            Uf2 = Normal(loc=tf.zeros([H1, H2]), scale=tf.ones([H1, H2]))
            Wi2 = Normal(loc=tf.zeros([H2, H2]), scale=tf.ones([H2, H2]))
            Ui2 = Normal(loc=tf.zeros([H1, H2]), scale=tf.ones([H1, H2]))
            Wo2 = Normal(loc=tf.zeros([H2, H2]), scale=tf.ones([H2, H2]))
            Uo2 = Normal(loc=tf.zeros([H1, H2]), scale=tf.ones([H1, H2]))
            Wc2 = Normal(loc=tf.zeros([H2, H2]), scale=tf.ones([H2, H2]))
            Uc2 = Normal(loc=tf.zeros([H1, H2]), scale=tf.ones([H1, H2]))

            Wy = Normal(loc=tf.zeros([H2, 1]), scale=tf.ones([H2, 1]))
            bf1 = Normal(loc=tf.zeros(H1), scale=tf.ones(H1))
            bi1 = Normal(loc=tf.zeros(H1), scale=tf.ones(H1))
            bo1 = Normal(loc=tf.zeros(H1), scale=tf.ones(H1))
            bc1 = Normal(loc=tf.zeros(H1), scale=tf.ones(H1))
            bf2 = Normal(loc=tf.zeros(H2), scale=tf.ones(H2))
            bi2 = Normal(loc=tf.zeros(H2), scale=tf.ones(H2))
            bo2 = Normal(loc=tf.zeros(H2), scale=tf.ones(H2))
            bc2 = Normal(loc=tf.zeros(H2), scale=tf.ones(H2))
            by = Normal(loc=tf.zeros(1), scale=tf.ones(1))

            X = tf.placeholder(tf.float32, [N, sequence_length, D])
            y = Normal(loc=two_LSTM_layer(X, Wf1, Uf1, Wi1, Ui1, Wo1, Uo1,
                                          Wc1, Uc1, bf1, bi1, bo1, bc1, Wf2,
                                          Uf2, Wi2, Ui2, Wo2, Uo2, Wc2, Uc2,
                                          bf2, bi2, bo2, bc2, Wy, by, H1, H2),
                       scale=1.)
            # BACKWARD MODEL A
            q_Wf1 = Normal(loc=tf.Variable(tf.random_normal([H1, H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1, H1]))))
            q_Uf1 = Normal(loc=tf.Variable(tf.random_normal([D, H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H1]))))
            q_Wi1 = Normal(loc=tf.Variable(tf.random_normal([H1, H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1, H1]))))
            q_Ui1 = Normal(loc=tf.Variable(tf.random_normal([D, H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H1]))))
            q_Wo1 = Normal(loc=tf.Variable(tf.random_normal([H1, H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1, H1]))))
            q_Uo1 = Normal(loc=tf.Variable(tf.random_normal([D, H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H1]))))
            q_Wc1 = Normal(loc=tf.Variable(tf.random_normal([H1, H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1, H1]))))
            q_Uc1 = Normal(loc=tf.Variable(tf.random_normal([D, H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H1]))))

            q_Wf2 = Normal(loc=tf.Variable(tf.random_normal([H2, H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2, H2]))))
            q_Uf2 = Normal(loc=tf.Variable(tf.random_normal([H1, H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1, H2]))))
            q_Wi2 = Normal(loc=tf.Variable(tf.random_normal([H2, H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2, H2]))))
            q_Ui2 = Normal(loc=tf.Variable(tf.random_normal([H1, H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1, H2]))))
            q_Wo2 = Normal(loc=tf.Variable(tf.random_normal([H2, H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2, H2]))))
            q_Uo2 = Normal(loc=tf.Variable(tf.random_normal([H1, H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1, H2]))))
            q_Wc2 = Normal(loc=tf.Variable(tf.random_normal([H2, H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2, H2]))))
            q_Uc2 = Normal(loc=tf.Variable(tf.random_normal([H1, H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1, H2]))))

            q_Wy = Normal(loc=tf.Variable(tf.random_normal([H2, 1])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2, 1]))))
            q_bf1 = Normal(loc=tf.Variable(tf.random_normal([H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1]))))
            q_bi1 = Normal(loc=tf.Variable(tf.random_normal([H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1]))))
            q_bo1 = Normal(loc=tf.Variable(tf.random_normal([H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1]))))
            q_bc1 = Normal(loc=tf.Variable(tf.random_normal([H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1]))))
            q_bf2 = Normal(loc=tf.Variable(tf.random_normal([H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2]))))
            q_bi2 = Normal(loc=tf.Variable(tf.random_normal([H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2]))))
            q_bo2 = Normal(loc=tf.Variable(tf.random_normal([H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2]))))
            q_bc2 = Normal(loc=tf.Variable(tf.random_normal([H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2]))))
            q_by = Normal(loc=tf.Variable(tf.random_normal([1])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([1]))))
            # INFERENCE A
            # this will take a couple of minutes
            inference = ed.KLqp(latent_vars={Wf1: q_Wf1, Uf1: q_Uf1,
                                             Wi1: q_Wi1, Ui1: q_Ui1,
                                             Wo1: q_Wo1, Uo1: q_Uo1,
                                             Wc1: q_Wc1, Uc1: q_Uc1,
                                             Wf2: q_Wf2, Uf2: q_Uf2,
                                             Wi2: q_Wi2, Ui2: q_Ui2,
                                             Wo2: q_Wo2, Uo2: q_Uo2,
                                             Wc2: q_Wc2, Uc2: q_Uc2,
                                             bf1: q_bf1, bi1: q_bi1,
                                             bo1: q_bo1, bc1: q_bc1,
                                             bf2: q_bf2, bi2: q_bi2,
                                             bo2: q_bo2, bc2: q_bc2,
                                             Wy: q_Wy, by: q_by},
                                data={X: seq_array1, y: label_array1})
            inference.run(n_samples=5, n_iter=2500)
            Xp = tf.placeholder(tf.float32, seq_array2.shape)
            y_preds = [sess.run(two_LSTM_layer(Xp, q_Wf1, q_Uf1, q_Wi1,
                                               q_Ui1, q_Wo1, q_Uo1, q_Wc1,
                                               q_Uc1, q_bf1, q_bi1, q_bo1,
                                               q_bc1, q_Wf2, q_Uf2, q_Wi2,
                                               q_Ui2, q_Wo2, q_Uo2, q_Wc2,
                                               q_Uc2, q_bf2, q_bi2, q_bo2,
                                               q_bc2, q_Wy, q_by, H1, H2),
                                {Xp: seq_array2}) for _ in range(50)]

        elif model_name == 'Two Layer GRU':

            Wz1 = Normal(loc=tf.zeros([H1, H1]), scale=tf.ones([H1, H1]))
            Uz1 = Normal(loc=tf.zeros([D, H1]), scale=tf.ones([D, H1]))
            Wr1 = Normal(loc=tf.zeros([H1, H1]), scale=tf.ones([H1, H1]))
            Ur1 = Normal(loc=tf.zeros([D, H1]), scale=tf.ones([D, H1]))
            Wh1 = Normal(loc=tf.zeros([H1, H1]), scale=tf.ones([H1, H1]))
            Uh1 = Normal(loc=tf.zeros([D, H1]), scale=tf.ones([D, H1]))

            Wz2 = Normal(loc=tf.zeros([H2, H2]), scale=tf.ones([H2, H2]))
            Uz2 = Normal(loc=tf.zeros([H1, H2]), scale=tf.ones([H1, H2]))
            Wr2 = Normal(loc=tf.zeros([H2, H2]), scale=tf.ones([H2, H2]))
            Ur2 = Normal(loc=tf.zeros([H1, H2]), scale=tf.ones([H1, H2]))
            Wh2 = Normal(loc=tf.zeros([H2, H2]), scale=tf.ones([H2, H2]))
            Uh2 = Normal(loc=tf.zeros([H1, H2]), scale=tf.ones([H1, H2]))

            Wy = Normal(loc=tf.zeros([H2, 1]), scale=tf.ones([H2, 1]))
            bz1 = Normal(loc=tf.zeros(H1), scale=tf.ones(H1))
            br1 = Normal(loc=tf.zeros(H1), scale=tf.ones(H1))
            bh1 = Normal(loc=tf.zeros(H1), scale=tf.ones(H1))
            bz2 = Normal(loc=tf.zeros(H2), scale=tf.ones(H2))
            br2 = Normal(loc=tf.zeros(H2), scale=tf.ones(H2))
            bh2 = Normal(loc=tf.zeros(H2), scale=tf.ones(H2))
            by = Normal(loc=tf.zeros(1), scale=tf.ones(1))

            X = tf.placeholder(tf.float32, [N, sequence_length, D])
            y = Normal(loc=two_GRU_layer(X, Wz1, Uz1, Wr1, Ur1, Wh1, Uh1,
                                         bz1, br1, bh1, Wz2, Uz2, Wr2,
                                         Ur2, Wh2, Uh2, bz2, br2, bh2,
                                         Wy, by, H1, H2), scale=1.)

            # BACKWARD MODEL A
            q_Wz1 = Normal(loc=tf.Variable(tf.random_normal([H1, H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1, H1]))))
            q_Uz1 = Normal(loc=tf.Variable(tf.random_normal([D, H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H1]))))
            q_Wr1 = Normal(loc=tf.Variable(tf.random_normal([H1, H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1, H1]))))
            q_Ur1 = Normal(loc=tf.Variable(tf.random_normal([D, H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H1]))))
            q_Wh1 = Normal(loc=tf.Variable(tf.random_normal([H1, H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1, H1]))))
            q_Uh1 = Normal(loc=tf.Variable(tf.random_normal([D, H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([D, H1]))))

            q_Wz2 = Normal(loc=tf.Variable(tf.random_normal([H2, H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2, H2]))))
            q_Uz2 = Normal(loc=tf.Variable(tf.random_normal([H1, H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1, H2]))))
            q_Wr2 = Normal(loc=tf.Variable(tf.random_normal([H2, H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2, H2]))))
            q_Ur2 = Normal(loc=tf.Variable(tf.random_normal([H1, H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1, H2]))))
            q_Wh2 = Normal(loc=tf.Variable(tf.random_normal([H2, H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2, H2]))))
            q_Uh2 = Normal(loc=tf.Variable(tf.random_normal([H1, H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1, H2]))))

            q_Wy = Normal(loc=tf.Variable(tf.random_normal([H2, 1])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2, 1]))))
            q_bz1 = Normal(loc=tf.Variable(tf.random_normal([H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1]))))
            q_br1 = Normal(loc=tf.Variable(tf.random_normal([H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1]))))
            q_bh1 = Normal(loc=tf.Variable(tf.random_normal([H1])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H1]))))
            q_bz2 = Normal(loc=tf.Variable(tf.random_normal([H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2]))))
            q_br2 = Normal(loc=tf.Variable(tf.random_normal([H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2]))))
            q_bh2 = Normal(loc=tf.Variable(tf.random_normal([H2])),
                           scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([H2]))))
            q_by = Normal(loc=tf.Variable(tf.random_normal([1])),
                          scale=tf.nn.softplus(
                tf.Variable(tf.random_normal([1]))))
            # INFERENCE A
            # this will take a couple of minutes
            inference = ed.KLqp(latent_vars={Wz1: q_Wz1, Uz1: q_Uz1,
                                             Wr1: q_Wr1, Ur1: q_Ur1,
                                             Wh1: q_Wh1, Uh1: q_Uh1,
                                             Wz2: q_Wz2, Uz2: q_Uz2,
                                             Wr2: q_Wr2, Ur2: q_Ur2,
                                             Wh2: q_Wh2, Uh2: q_Uh2,
                                             bz1: q_bz1, bz1: q_bz1,
                                             bh1: q_bh1, bz2: q_bz2,
                                             bz2: q_bz2, bh2: q_bh2,
                                             Wy: q_Wy, by: q_by},
                                data={X: seq_array1,
                                      y: label_array1})
            inference.run(n_samples=5, n_iter=2500)
            Xp = tf.placeholder(tf.float32, seq_array2.shape)
            y_preds = [sess.run(two_GRU_layer(Xp, q_Wz1, q_Uz1,
                                              q_Wr1, q_Ur1, q_Wh1, q_Uh1,
                                              q_bz1, q_br1, q_bh1,
                                              q_Wz2, q_Uz2, q_Wr2, q_Ur2,
                                              q_Wh2, q_Uh2, q_bz2,
                                              q_br2, q_bh2, q_Wy, q_by,
                                              H1, H2),
                                {Xp: seq_array2}) for _ in range(50)]
        else:
            raise Exception('Please specify a valid model!')
        # Critisism
        # Histogram
        sns.distplot(y_preds[0])
        plt.title('Histogram of RUL, Bayesian {}'.format(model_name))
        plt.xlabel('RUL')
        plt.show()
        # RMSE
        print('Average Validation RMSE: {}'.format(np.mean([np.sqrt( \
          mean_squared_error(label_array2,y_pred)) for y_pred in y_preds])))
        print('Average Validation RMSE for RUL under 50: {}'.format(
                np.mean([np.sqrt(mean_squared_error(label_array2[less_50],
                                                    y_pred[less_50]))
                     for y_pred in y_preds])))
        # Prediction time series
        pd.DataFrame([label_array2, y_preds[0]]).transpose().rename(
            columns={0: 'True', 1: 'Pred'})[-1500:].plot()
        plt.title('Prediction of RUL, Bayesian {}'.format(model_name))
        plt.xlabel('RUL')
        plt.show()
        # Posterior prediction distribution 1500
        [plt.plot(y_pred[-1500:], color='black', alpha=0.1)
         for y_pred in y_preds]
        plt.plot(label_array2[-1500:])
        plt.title('Distribution of Prediction of RUL, Bayesian {}'.format(model_name))
        plt.xlabel('RUL(last 1500 days)')
        plt.show()
        # Posterior prediction distribution 150
        [plt.plot(y_pred[-150:], color='black', alpha=0.1)
         for y_pred in y_preds]
        plt.plot(label_array2[-150:])
        plt.title('Distribution of Prediction of RUL, Bayesian {}'.format(model_name))
        plt.xlabel('RUL(last 150 days)')
        plt.show()
    else:
        raise Exception(
            'Please specify a boolean for use Bayesian inference or not!')
