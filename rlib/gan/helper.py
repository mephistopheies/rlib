import numpy as np
from lasagne.utils import floatX


def sample_uniform_noise(n, d):
    return floatX(np.random.uniform(size=(n, d)))


def sample_normal_noise(n, d, m=0, s=1):
    return floatX(np.random.normal(m, s, size=(n, d)))
