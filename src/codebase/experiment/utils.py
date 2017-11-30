import tensorflow as tf
import numpy as np
from tensorflow.contrib.distributions import Normal


def export_state_tuples(state_tuples, name):
    for state_tuple in state_tuples:
        tf.add_to_collection(name, state_tuple.c)
        tf.add_to_collection(name, state_tuple.h)


def with_prefix(prefix, name):
    """Adds prefix to name."""
    return "/".join((prefix, name))


def import_state_tuples(state_tuples, name, num_replicas):
    restored = []
    for i in range(len(state_tuples) * num_replicas):
        c = tf.get_collection_ref(name)[2 * i + 0]
        h = tf.get_collection_ref(name)[2 * i + 1]
        restored.append(tf.contrib.rnn.LSTMStateTuple(c, h))
    return tuple(restored)


def log_gaussian(x, mu, sigma):
    return -.5 * np.log(2. * np.pi) - tf.log(tf.abs(sigma)) - \
        (x - mu) ** 2 / (2. * sigma ** 2)


def log_gaussian_(x, mu, rho):
    return -.5 * np.log(2. * np.pi) - rho / 2. - (x - mu) ** 2 / \
        (2. * tf.exp(rho))


def KL_scale_mixture(shape, mu, sigma, prior, w):
    """Compute KL for scale mixture Gaussian priors
    shape = (n_unit, n_w)
    """
    posterior = Normal(mu, sigma)
    part_post = posterior.log_prob(tf.reshape(w, [-1]))  # flatten
    prior_1 = Normal(0., prior.sigma_1)
    prior_2 = Normal(0., prior.sigma_2)
    part_1 = tf.reduce_sum(prior_1.log_prob(w)) + tf.log(prior.pi)
    part_2 = tf.reduce_sum(prior_2.log_prob(w)) + tf.log(prior.pi)
    prior_mix = tf.stack([part_1, part_2])
    KL = - tf.reduce_sum(tf.reduce_logsumexp(prior_mix, axis=0)) + \
        tf.reduce_sum(part_post)
    return KL
