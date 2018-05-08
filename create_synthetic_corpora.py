import numpy as np
from utils import math_utils
import itertools as itr
import pickle

K = 10
T = 12
V = 1000
D = np.random.randint(10 - 2, 10 + 2, size=T)
N = [[np.random.randint(100 - 5, 100 + 5) for d in range(D[t])]for t in range(T)]

ALPHA_VAR = 1  # SIGMA in paper
ETA_VAR = 1  # PSI in paper
PHI_VAR = 1  # BETA in paper


def simulate_corpus(i):
    print('generating alphas')
    alphas = np.zeros([T, K])
    alphas[0] = np.random.normal(0, ALPHA_VAR, size=K)
    for t in range(1, T):
        alphas[t] = np.random.normal(alphas[t], ALPHA_VAR, size=K)

    print('generating etas')
    etas = [np.zeros([D[t], K]) for t in range(T)]
    for t in range(T):
        for d in range(D[t]):
            etas[t][d] = np.random.normal(alphas[t], ETA_VAR, size=K)

    print('generating phis')
    phis = np.zeros([T, K, V])
    softmax_phis = np.zeros([T, K, V])
    phis[0] = np.random.normal(0, PHI_VAR, size=[K, V])
    for t in range(1, T):
        for k in range(K):
            phis[t, k] = np.random.normal(phis[t-1, k], PHI_VAR, size=V)
    for t, k in itr.product(range(T), range(K)):
        softmax_phis[t, k] = math_utils.softmax_(phis[t, k])

    print('generating zs')
    zs = [[np.zeros([N[t][d]], dtype=int) for d in range(D[t])] for t in range(T)]
    for t in range(T):
        for d in range(D[t]):
            ps = math_utils.softmax_(etas[t][d])
            sm = np.sum(ps)
            if sm != 1:
                ps[-1] += 1 - sm
            s = np.random.choice(range(K), p=ps, size=N[t][d])
            zs[t][d] = s

    print('generating ws')
    ws = [[np.zeros(N[t][d], dtype=int) for d in range(D[t])] for t in range(T)]
    for t in range(T):
        for d in range(D[t]):
            for n in range(N[t][d]):
                k = zs[t][d][n]
                ps = softmax_phis[t, k]
                sm = np.sum(ps)
                if sm != 1:
                    ps[np.argmax(ps)] += 1 - sm
                w = np.random.choice(range(V), p=ps)
                ws[t][d][n] = w

    vars = {
        'alphas': alphas,
        'etas': etas,
        'phis': phis,
        'zs': zs,
        'ws': ws
    }
    pickle.dump(vars, open('data/sim_corpora/corpus%d.p' % i, 'wb'))


for i in range(100):
    print("==== CORPUS %d ====" % i)
    simulate_corpus(i)

first_corp = pickle.load(open('data/sim_corpora/corpus0.p', 'rb'))

