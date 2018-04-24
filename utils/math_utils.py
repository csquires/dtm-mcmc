import numpy as np
from scipy.special import logsumexp
from sklearn.utils.extmath import softmax


def softmax_(x):
    if x.ndim == 1:
        return softmax(x.reshape(1, -1)).squeeze()
    else:
        return softmax(x)


def coin(p):
    return np.random.binomial(1, p)


def complete_square(normals):
    """
    Compute mean and covariance of normal distribution resulting from multiplication of other normals (each
    with diagonal, homoscedastic covariance matrix)

    :param normals: list of (mean, variance) of multiplied normal distributions
    :return:
    """
    cov = sum([1/n[1] for n in normals]) ** -1
    mean = cov * np.array([n[0]/n[1] for n in normals]).sum(axis=0)
    return mean, cov


def get_alias_table(ps):
    n = len(ps)
    alias_table = [None]*n
    prob_table = [None]*n

    small = []
    large = []
    ps_scaled = []
    for i, p in enumerate(ps):
        p_scaled = p*n
        ps_scaled.append(p_scaled)
        if p_scaled >= 1:
            large.append(i)
        else:
            small.append(i)

    while small and large:
        l = small.pop()
        g = large.pop()
        prob_table[l] = ps_scaled[l]
        alias_table[l] = g
        pg = ps_scaled[g] - (1 - ps_scaled[l])
        ps_scaled[g] = pg
        if pg < 1:
            small.append(g)
        else:
            large.append(g)

    while large:
        g = large.pop()
        prob_table[g] = 1

    while small:
        l = small.pop()
        prob_table[l] = 1

    return prob_table, alias_table


def sample_alias_table(alias_table, prob_table, size=1):
    n = len(alias_table)
    if size == 1:
        i = np.random.randint(0, n)
        p = prob_table[i]
        if p != 1:
            return i if coin(p) else alias_table[i]
        else:
            return i
    else:
        ixs = np.random.randint(0, n, size=size)
        samples = np.zeros(size, dtype=int)
        for s, ix in enumerate(ixs):
            p = prob_table[ix]
            if p != 1:
                samples[s] = ix if coin(p) else alias_table[ix]
            else:
                samples[s] = ix
        return samples





