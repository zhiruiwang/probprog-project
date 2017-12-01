from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import math
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.distributions import Normal

from data_preprocessing import load_and_preprocess
from data_preprocessing import config_data
from utils import export_state_tuples
from utils import with_prefix
from utils import import_state_tuples
from utils import KL_scale_mixture

# Parse input:
parser = argparse.ArgumentParser()

parser.add_argument('-download', action='store_true', help='Download dataset.')
parser.add_argument('-save_path', type=str, default='./model/saved_new')
parser.add_argument('-layers', type=list, default=[100, 50])
parser.add_argument('-seq_length', type=int, default=5)
parser.add_argument('-batch_size', type=int, default=200)
parser.add_argument('-n_epoch', type=int, default=30)
parser.add_argument('-learning_rate', type=float, default=0.01)
parser.add_argument('-lr_decay', type=float, default=0.9)
parser.add_argument('-n_epoch_decay', type=int, default=15)
parser.add_argument('-bias', action='store_true', help='Enable biases.')
parser.add_argument('-pi', type=float, default=0.5)
parser.add_argument('-rho_1', type=float, default=1.0)
parser.add_argument('-rho_2', type=float, default=0.5)
parser.add_argument('-inference_mode', type=str,
                    default='mu', choices=['mu', 'sample'])
parser.add_argument('-random_seed', type=int, default=12)

input_args = parser.parse_args()


# Global Parameters

# Parameters for operations
download = input_args.download
save_path = input_args.save_path
random_seed = input_args.random_seed

# Parameters for priors
pi = input_args.pi
rho_1 = input_args.rho_1
rho_2 = input_args.rho_2
sigma_1 = tf.nn.softplus(rho_1) + 1e-8
sigma_2 = tf.nn.softplus(rho_2) + 1e-8
sigma_mix = np.sqrt(0.5 * np.square(1.) + (1. - 0.5) * np.square(0.5))

# Parameters for network structure
layers = input_args.layers
n_layer = len(layers)
batch_size = input_args.batch_size
seq_length = input_args.seq_length
n_epoch = input_args.n_epoch
n_feature = 25
inference_mode = input_args.inference_mode
bias = input_args.bias

# Parameters for learning rate and optimization
init_scale = 0.5  # initial randomization scale
max_grad_norm = 5  # gradient clipping by a global norm
learning_rate = input_args.learning_rate
lr_decay = input_args.lr_decay
# learning rate decay after 'n_epoch_decay' epochs
n_epoch_decay = input_args.n_epoch_decay

print("Loading and preprocessing dataset...")

# Load data
train_df, test_df = load_and_preprocess(download=False)

# Build datasets based on the given configuration
X_train, y_train = config_data(train_df, seq_length, batch_size)
X_test, y_test = config_data(test_df, seq_length, batch_size)

n_batch_train = X_train.shape[0] // batch_size
n_batch_test = X_test.shape[0] // batch_size


def pm_producer(X, Y, batch_size, name=None):
    """Iterate on the raw data.

    This chunks up raw_data into batches of examples and returns Tensors that
    are drawn from these batches.

    Args:
    X, Y: the raw data.
    batch_size: int, the batch size.
    num_steps: int, the number of unrolls.
    name: the name of this operation (optional).

    Returns:
    A pair of Tensors:
    The first: each shaped [batch_size, seq_length, n_feature].
    The second element is the same data time-shifted to the right by one.
    """
    with tf.name_scope(name, "PMProducer", [X, Y, batch_size]):
        X = tf.convert_to_tensor(X, name="X", dtype=tf.float32)
        Y = tf.convert_to_tensor(Y, name="Y", dtype=tf.float32)

    data_len = tf.size(Y)
    batch_len = data_len // batch_size
    X = tf.slice(X, [0, 0, 0], [batch_size*batch_len, -1, -1])
    Y = tf.slice(Y, [0], [batch_size*batch_len])

    i = tf.train.range_input_producer(batch_len, shuffle=False).dequeue()
    x = tf.slice(X, [i * batch_size, 0, 0], [(i + 1) * batch_size, -1, -1])
    x.set_shape([batch_size, seq_length, n_feature])
    y = tf.slice(Y, [i * batch_size], [(i + 1) * batch_size])
    y.set_shape([batch_size])
    return x, y


class BayesianLSTM(BasicLSTMCell):
    """Build Bayesian LSTM layer with given configuration
    1. Inherit from tensorflow BasicLSTMCell
    2. Only difference in replacing with noisy weights
    """

    def __init__(self, n_unit_pre, n_unit, prior, is_training,
                 inference_mode, bias=True, forget_bias=1.0,
                 state_is_tuple=True, activation=tf.tanh,
                 reuse=None, name=None):
        super().__init__(n_unit, forget_bias, state_is_tuple, activation,
                         reuse=reuse)

        self.prior = prior
        self.bias = bias
        self.is_training = is_training
        self.n_unit = n_unit
        self.n_unit_pre = n_unit_pre
        self.inference_mode = inference_mode
        self.theta = None
        self.b = None
        self.name = name

    def _output(self, theta, b, inputs, h):
        xh = tf.concat([inputs, h], 1)
        return tf.matmul(xh, theta) + tf.squeeze(b)

    def call(self, inputs, state):
        """Long short-term memory cell (LSTM).
        Args:
         inputs: `2-D` tensor with shape `[batch_size x input_size]`.
         state: An `LSTMStateTuple` of state tensors, each shaped
            `[batch_size x self.state_size]`, if `state_is_tuple` is set to
            `True`. Otherwise, a `Tensor` shaped
            `[batch_size x 2 * self.state_size]`.
        Returns:
         A pair containing the new hidden state, and the new state (either a
            `LSTMStateTuple` or a concatenated state, depending on
            `state_is_tuple`).
        """
        # get noisy weights
        if self.theta is None:
            # Fetch initialization params from prior
            rho_min_init, rho_max_init = self.prior.lstm_init()
            self.theta = get_noisy_weights(shape=(self.n_unit_pre +
                                                  self.n_unit,
                                                  4 * self.n_unit),
                                           name=self.name + '_theta',
                                           prior=self.prior,
                                           is_training=self.is_training,
                                           rho_min_init=rho_min_init,
                                           rho_max_init=rho_max_init)
            if self.bias:
                self.b = get_noisy_weights(shape=(4 * self.n_unit, 1),
                                           name=self.name + '_b',
                                           prior=self.prior,
                                           is_training=self.is_training,
                                           rho_min_init=rho_min_init,
                                           rho_max_init=rho_max_init)
            else:
                self.b = tf.get_variable(self.name + '_b',
                                         (4 * self.n_unit, 1),
                                         tf.float32,
                                         tf.constant_initializer(0.))

        # Parameters of gates are concatenated into one multiply for
        # efficiency.
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=1)

        concat = self._output(self.theta, self.b, inputs, h)

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)

        new_c = (
            c * tf.sigmoid(f + self._forget_bias) +
            tf.sigmoid(i) * self._activation(j))
        new_h = self._activation(new_c) * tf.sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(c=new_c, h=new_h)
        else:
            new_state = tf.concat(values=[new_c, new_h], axis=1)

        return new_h, new_state


class ScaleMixturePrior:
    """Build scale mixture prior with given configuration
    1. A softplus is on top of the given rho's.
    2. Restrict sample min max to prevent extreme values.
    """
    # zero mean, mixture of two variances

    def __init__(self):
        self.pi = pi
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        self.sigma_1 = tf.nn.softplus(rho_1) + 1e-8
        self.sigma_2 = tf.nn.softplus(rho_2) + 1e-8
        self.sigma_mix = np.sqrt(0.5 * np.square(1.) +
                                 (1. - 0.5) * np.square(0.5))

    def lstm_init(self):
        rho_max_init = math.log(math.exp(self.sigma_mix / 2.) - 1.)
        rho_min_init = math.log(math.exp(self.sigma_mix / 4.) - 1.)
        return rho_min_init, rho_max_init

    def normal_init(self):
        rho_max_init = math.log(math.exp(self.sigma_mix / 1.) - 1.)
        rho_min_init = math.log(math.exp(self.sigma_mix / 2.) - 1.)
        return rho_min_init, rho_max_init


def get_noisy_weights(shape, name, prior, is_training, rho_min_init=None,
                      rho_max_init=None):
    """Get noisy weights
    1. Sample weights as given shape and configuration
    2. Update histogram summary
    3. Update KLqp
    4. Return distribution of weights variables
    """
    # add mean
    with tf.variable_scope('BBB', reuse=not is_training):
        mu = tf.get_variable(name + '_mean', shape, dtype=tf.float32)

    # add rho
    if rho_min_init is None or rho_max_init is None:
        rho_min_init, rho_max_init = prior.lstm_init()
    rho_init = tf.random_uniform_initializer(rho_min_init, rho_max_init)
    with tf.variable_scope('BBB', reuse=not is_training):
        rho = tf.get_variable(name + '_rho', shape, dtype=tf.float32,
                              initializer=rho_init)

    # control output
    if is_training or inference_mode == 'sample':
        epsilon = Normal(0., 1.).sample(shape)
        sigma = tf.nn.softplus(rho) + 1e-8
        w = mu + sigma * epsilon
    else:
        w = mu

    if is_training:
        return w

    # create histogram
    tf.summary.histogram(name + '_mu_hist', mu)
    tf.summary.histogram(name + '_sigma_hist', sigma)
    tf.summary.histogram(name + '_rho_hist', rho)

    # KL
    kl = KL_scale_mixture(shape,
                          tf.reshape(mu, [-1]),
                          tf.reshape(sigma, [-1]),
                          prior,
                          w)
    tf.add_to_collection('KL_layers', kl)

    return w


class BayesByBackprop(object):

    def __init__(self, is_training, X, y):
        self._is_training = is_training
        self._rnn_params = None
        self._cell = None
        self.batch_size = 200
        self.seq_length = 5
        self.X = X

        if is_training:
            n_batch = n_batch_train
        else:
            n_batch = n_batch_test

        # Construct prior
        prior = ScaleMixturePrior()

        n_unit_pre = n_feature
        # create 2 LSTMCells
        rnn_layers = []
        n_unit_pre = n_feature
        for i in range(n_layer):
            rnn_layers.append(BayesianLSTM(n_unit_pre, layers[i],
                                           prior, is_training,
                                           inference_mode=inference_mode,
                                           forget_bias=0.0,
                                           name='bbb_lstm_{}'.format(i),
                                           bias=True))
            n_unit_pre = layers[i]

        multi_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_layers)
        self._initial_state = multi_rnn_cell.zero_state(batch_size, tf.float32)
        state = self._initial_state

        # 'output' is a tensor of shape [batch_size, seq_length, n_feature]
        # 'state' is a N-tuple where N is the number of LSTMCells containing a
        # tf.contrib.rnn.LSTMStateTuple for each cell
        outputs, state = tf.nn.dynamic_rnn(cell=multi_rnn_cell,
                                           inputs=X,
                                           time_major=False,
                                           dtype=tf.float32)

        # output layer
        # add weight term
        rho_min_init, rho_max_init = prior.normal_init()
        if bias:
            w = get_noisy_weights((50, 1), 'w', prior,
                                  is_training, rho_min_init, rho_max_init)
        else:
            w = tf.get_variable('w', (50, 1), tf.float32,
                                tf.constant_initializer(0.))

        # add bias term
        if bias:
            b = get_noisy_weights(
                (1), 'b', prior, is_training, rho_min_init, rho_max_init)
        else:
            b = tf.get_variable('b', (1), tf.float32,
                                tf.constant_initializer(0.))

        output = tf.reshape(
            tf.matmul(outputs[:, seq_length-1, :], w) + b, [-1])

        y = tf.reshape(y, [-1])
        y_pred = Normal(output, 1.)
        print("Finish predicting y")

        # Use the contrib sequence loss and average over the batches
        loss = - tf.log(y_pred.prob(y) + 1e-8)

        # Update the cost
        self._cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state

        # 1. For testing, no kl term, just loss
        self._kl_div = 0.
        if not is_training:
            return

        # 2. For training, compute kl scaled by 1./n_batch
        # Add up all prior's kl values
        kl_div = tf.add_n(tf.get_collection('KL_layers'), 'kl_divergence')
        # Compute ELBO
        kl_const = 1. / n_batch
        self._kl_div = kl_div * kl_const
        self._total_loss = self._cost + self._kl_div
        # Optimization:
        # Learning rate
        self._lr = tf.Variable(0.0, trainable=False)
        # Update all weights with gradients
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._total_loss,
                                                       tvars),
                                          max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())
        # Learning rate update
        self._new_lr = tf.placeholder(
            tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        print("Finish building model")

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    def export_ops(self, name):
        """Exports ops to collections."""
        self._name = name
        ops = {with_prefix(self._name, "cost"): self._cost,
               with_prefix(self._name, "kl_div"): self._kl_div}
        if self._is_training:
            ops.update(lr=self._lr, new_lr=self._new_lr,
                       lr_update=self._lr_update)
            if self._rnn_params:
                ops.update(rnn_params=self._rnn_params)
        for name, op in ops.items():
            tf.add_to_collection(name, op)
        self._initial_state_name = with_prefix(self._name, "initial")
        self._final_state_name = with_prefix(self._name, "final")
        export_state_tuples(self._initial_state, self._initial_state_name)
        export_state_tuples(self._final_state, self._final_state_name)

    def import_ops(self):
        """Imports ops from collections."""
        if self._is_training:
            self._train_op = tf.get_collection_ref("train_op")[0]
            self._lr = tf.get_collection_ref("lr")[0]
            self._new_lr = tf.get_collection_ref("new_lr")[0]
            self._lr_update = tf.get_collection_ref("lr_update")[0]
            rnn_params = tf.get_collection_ref("rnn_params")
            if self._cell and rnn_params:
                params_saveable = tf.contrib.cudnn_rnn.RNNParamsSaveable(
                    self._cell,
                    self._cell.params_to_canonical,
                    self._cell.canonical_to_params,
                    rnn_params,
                    base_variable_scope="Model/RNN")
                tf.add_to_collection(
                    tf.GraphKeys.SAVEABLE_OBJECTS, params_saveable)
        self._cost = tf.get_collection_ref(with_prefix(self._name, "cost"))[0]
        self._kl_div = tf.get_collection_ref(
            with_prefix(self._name, "kl_div"))[0]
        num_replicas = 1
        self._initial_state = import_state_tuples(
            self._initial_state, self._initial_state_name, num_replicas)
        self._final_state = import_state_tuples(
            self._final_state, self._final_state_name, num_replicas)

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name

    @property
    def kl_div(self):
        return self._kl_div if self._is_training else tf.constant(0.)


def run_epoch(session, model, eval_op=None, verbose=False):
    """Run the model on the given data
    """
    start_time = time.time()
    costs = 0.0
    state = session.run(model.initial_state)

    fetches = {
        "cost": model.cost,
        "final_state": model.final_state,
    }

    n_batch = n_batch_train

    if eval_op is not None:
        n_batch = n_batch_test
        fetches["eval_op"] = eval_op
        fetches["kl_divergence"] = model.kl_div

    for step in range(n_batch):
        feed_dict = {}
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        vals = session.run(fetches, feed_dict)
        cost = vals["cost"]
        state = vals["final_state"]

        costs += cost

        if verbose and (n_batch % 10 == 0):
            print("%.3f batch loss: %.3f speed: %.0f second per batch" %
                  (n_batch, cost, (time.time() - start_time)/n_batch)
                  )

            if model._is_training:
                print("KL is {}".format(vals["kl_divergence"]))

    normal_constant = .5 * np.log(2 * np.pi * sigma_mix ** 2)
    RMSE = np.sqrt((costs / n_batch - normal_constant) * sigma_mix ** 2)
    return RMSE


with tf.Graph().as_default():
    initializer = tf.random_uniform_initializer(-init_scale, init_scale)

    with tf.name_scope("Train"):
        with tf.variable_scope("Model", reuse=None,
                               initializer=initializer):
            X_train_batch, y_train_batch = pm_producer(
                X_train, y_train, batch_size)
            m = BayesByBackprop(
                is_training=True, X=X_train_batch, y=y_train_batch)
        tf.summary.scalar("Training_Loss", m.cost)
        tf.summary.scalar("Learning_Rate", m.lr)
    print("Finish building train model")

    with tf.name_scope("Test"):
        with tf.variable_scope("Model", reuse=True,
                               initializer=initializer):
            X_test_batch, y_test_batch = pm_producer(
                X_test, y_test, batch_size)
            mtest = BayesByBackprop(
                is_training=False, X=X_test_batch, y=y_test_batch)
    print("Finish building test model")

    models = {"Train": m, "Test": mtest}
    for name, model in models.items():
        model.export_ops(name)
    metagraph = tf.train.export_meta_graph()
    soft_placement = False


with tf.Graph().as_default():
    tf.train.import_meta_graph(metagraph)
    for model in models.values():
        model.import_ops()
    sv = tf.train.Supervisor(logdir=save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
    with sv.managed_session(config=config_proto) as session:
        for i in range(n_epoch):
            lr_decay = lr_decay ** max(i + 1 - n_epoch_decay, 0.0)
            m.assign_lr(session, learning_rate * lr_decay)
            print("Epoch: %d Learning rate: %.3f" %
                  (i + 1, session.run(m.lr)))
            train_RMSE = run_epoch(
                session, m, eval_op=m.train_op, verbose=True)
            print("Epoch: %d Train loss: %.3f" % (i + 1, train_RMSE))
        test_RMSE = run_epoch(session, mtest, eval_op=None)
        print("Test loss: %.3f" % test_RMSE)
        if save_path:
            print("Saving model to %s." % save_path)
            sv.saver.save(session, save_path, global_step=sv.global_step)
