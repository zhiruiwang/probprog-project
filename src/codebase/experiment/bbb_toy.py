import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from utils import log_gaussian, log_gaussian_

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# DataSet class with practical operations for neural networks
print(mnist.train.images.shape)
print(mnist.test.images.shape)
print(mnist.validation.images.shape)

N = 55000  # number of data points
M = 100  # batch size during training
n_batch = int(N / M)  # number of batches
H = 200  # hidden layer size
D = 784  # data dimension
K = 10  # number of clusters

sigma = 0.1  # standard deviation of network parameters
n_epoch = 2000
learning_rate = 0.5
n_sample = 5
sigma_prior = tf.exp(-3.0)

# Placeholders for inputs
x = tf.placeholder(tf.float32, shape=None, name='x')
y = tf.placeholder(tf.float32, shape=None, name='y')

# Model Parameters
# Layer 1
W1_mu = tf.Variable(tf.truncated_normal([D, H], stddev=sigma))
W1_rho = tf.Variable(tf.truncated_normal([D, H], mean=1.0, stddev=sigma))
b1_mu = tf.Variable(tf.zeros([H]))
b1_rho = tf.Variable(tf.zeros([H]))

# Layer 2
W2_mu = tf.Variable(tf.truncated_normal([H, H], stddev=sigma))
W2_rho = tf.Variable(tf.truncated_normal([H, H], mean=1.0, stddev=sigma))
b2_mu = tf.Variable(tf.zeros([H]))
b2_rho = tf.Variable(tf.zeros([H]))

# Layer 3
W3_mu = tf.Variable(tf.truncated_normal([H, K], stddev=sigma))
W3_rho = tf.Variable(tf.truncated_normal([H, K], mean=1.0, stddev=sigma))
b3_mu = tf.Variable(tf.zeros([K]))
b3_rho = tf.Variable(tf.zeros([K]))

# Bayes by Backprop

# Step 1: Initialization
log_pw, log_qw, log_pD = 0., 0., 0.

# Step 2: Begin sampling
for _ in range(n_sample):

    # Sample epsilons
    epsilon_W1 = tf.random_normal(shape=[D, H], mean=0., stddev=1.)
    epsilon_b1 = tf.random_normal(shape=[H], mean=0., stddev=1.)
    epsilon_W2 = tf.random_normal(shape=[H, H], mean=0., stddev=1.)
    epsilon_b2 = tf.random_normal(shape=[H], mean=0., stddev=1.)
    epsilon_W3 = tf.random_normal(shape=[H, K], mean=0., stddev=1.)
    epsilon_b3 = tf.random_normal(shape=[K], mean=0., stddev=1.)

    # Compute weights
    W1 = W1_mu + tf.multiply(tf.log(1. + tf.exp(W1_rho)), epsilon_W1)
    b1 = b1_mu + tf.multiply(tf.log(1. + tf.exp(b1_rho)), epsilon_b1)
    W2 = W2_mu + tf.multiply(tf.log(1. + tf.exp(W2_rho)), epsilon_W2)
    b2 = b2_mu + tf.multiply(tf.log(1. + tf.exp(b2_rho)), epsilon_b2)
    W3 = W3_mu + tf.multiply(tf.log(1. + tf.exp(W3_rho)), epsilon_W3)
    b3 = b3_mu + tf.multiply(tf.log(1. + tf.exp(b3_rho)), epsilon_b3)

    # Build neural network
    h = tf.nn.relu(tf.matmul(x, W1) + b1)
    h = tf.nn.relu(tf.matmul(h, W2) + b2)
    h = tf.nn.softmax(tf.nn.relu(tf.matmul(h, W3) + b3))

    # Calculate log probabilities
    for W, b, W_mu, W_rho, b_mu, b_rho in [
            (W1, b1, W1_mu, W1_rho, b1_mu, b1_rho),
            (W2, b2, W2_mu, W2_rho, b2_mu, b2_rho),
            (W3, b3, W3_mu, W3_rho, b3_mu, b3_rho)]:
        # Calculate log priors
        log_pw = log_pw + tf.reduce_sum(log_gaussian(W, 0., 1.)) + \
            tf.reduce_sum(log_gaussian(b, 0., 1.))
        # Calculate log variational distributions
        log_qw = log_qw + tf.reduce_sum(log_gaussian_(W, W_mu, W_rho * 2)) + \
            tf.reduce_sum(log_gaussian_(b, b_mu, b_rho * 2))
    # Calculate log likelihood
    log_pD = tf.reduce_sum(log_gaussian(y, h, 1.))

# Calculate log probabilities over samples
log_qw = log_qw / n_sample
log_pw = log_pw / n_sample
log_pD = log_pD / n_sample

current_batch = tf.placeholder(tf.float32, shape=None, name='i')
pi = (2 ** (n_batch - current_batch - 1)) / (2 ** n_batch - 1)
KL = tf.reduce_sum(pi * (log_qw - log_pw) - log_pD) / float(M)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(KL)

h_mu = tf.nn.relu(tf.matmul(x, W1_mu) + b1_mu)
h_mu = tf.nn.relu(tf.matmul(h_mu, W2_mu) + b2_mu)
h_mu = tf.nn.softmax(tf.nn.relu(tf.matmul(h_mu, W3_mu) + b3_mu))

correct_prediction = tf.equal(tf.argmax(h_mu, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for n in range(n_epoch):
        KL_list = []
        pi_list = []
        for i in range(n_batch):
            batch = mnist.train.next_batch(M)
            results = sess.run([KL, train_step, pi, W2_rho],
                               feed_dict={x: batch[0],
                                          y: batch[1],
                                          current_batch: i})
            KL_list.append(results[0])
            pi_list.append(np.mean(results[3]))
        if n % 2 == 0:
            print("Epoch:", '%04d' %
                  (n + 1), "cost=", "{:.9f}".format(results[0]))
        print('test accuracy %g' % accuracy.eval(feed_dict={
            x: mnist.test.images, y: mnist.test.labels, current_batch: i}))
