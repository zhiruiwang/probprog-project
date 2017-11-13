import tensorflow as tf
import numpy as np

def log_gaussian(x, mu, sigma):
    return -.5 * np.log(2. * np.pi) - tf.log(tf.abs(sigma)) - (x - mu) ** 2 / (2. * sigma ** 2)

def log_gaussian_(x, mu, rho):
    return -.5 * np.log(2. * np.pi) - rho / 2. - (x - mu) ** 2 / (2. * tf.exp(rho))