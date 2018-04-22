import numpy as np
from utils import math_utils
from sklearn.utils.extmath import softmax

V = 1000  # vocab size

SIGMA = 1
PSI = 1
BETA = 1
EPS_A = 10
EPS_B = 1
EPS_C = 1.5


class DynamicTopicModel:
    def __init__(self, corpus, K):
        T = len(corpus)
        D = [len(timeslice) for timeslice in corpus]
        N = [[len(doc) for doc in timeslice] for timeslice in corpus]
        self.T = T
        self.D = D
        self.N = N
        self.K = K
        self.corpus = corpus
        self.alpha = np.zeros([T, K])  # overall topic mixture at time t
        self.eta = [np.zeros([D[t], K]) for t in range(T)]  # d-th document topic mixture at time t
        self.psi = np.zeros([T, K, V])  # k-th topic at time t (as mixture of V words)
        self.z = [np.zeros(N[t][d]) for t in range(T) for d in range(D[t])]
        self.topic_count_by_doc = [np.zeros(K, dtype=int) for t in range(T) for d in range(D[t])]
        self.topic_count_by_time = np.zeros([T, K], dtype=int)
        self.topic_count_by_word = np.zeros([T, K, V], dtype=int)
        self.nsamples_alias = np.zeros([T, V], dtype=int)
        self.BURN_IN = 100
        self.WAIT = 10
        self.alias_tables = [None for t in range(T) for w in range(V)]
        self.prob_tables = [None for t in range(T) for w in range(V)]

    def initialize(self):
        pass

    def sample(self, n_samples, verbose=False):
        samples = []
        for i in range(self.BURN_IN + self.WAIT * n_samples):
            if verbose:
                print('Iteration %d' % i)
            alpha, eta, psi, z = self.gibbs_iter(i)
            if i > self.BURN_IN and (i - self.BURN_IN) % self.WAIT == 0:
                samples.append({
                    'alpha': alpha,
                    'eta': eta,
                    'psi': psi,
                    'z': z
                })

    def update_alias(self, t, w):
        alias_table, prob_table = math_utils.get_alias_table(softmax(self.psi[t, :, w]))
        self.alias_tables[t][w] = alias_table
        self.prob_tables[t][w] = prob_table

    def sample_alias(self, t, w):
        alias_table = self.alias_tables[t][w]
        prob_table = self.prob_tables[t][w]
        return math_utils.sample_alias_table(alias_table, prob_table)

    def gibbs_iter(self, i):
        T = self.T
        D = self.D
        N = self.N
        K = self.K
        alpha = self.alpha
        eta = self.eta
        psi = self.psi
        z = self.z
        topic_count_by_doc = self.topic_count_by_doc
        topic_count_by_word = self.topic_count_by_word
        topic_count_by_time = self.topic_count_by_time
        nsamples_alias = self.nsamples_alias

        eps = EPS_A * (EPS_B + i) ** EPS_C
        xi = np.random.normal(0, eps ** 2, K)

        # Update alpha: closed form sample from normal
        old_alpha = alpha.copy()
        for t in range(T):
            eta_mean = eta[t].mean(axis=0)  # in R^K
            eta_mean_var = (eta_mean, PSI ** 2 / D[t])
            if t == 0:
                alpha_mean, alpha_cov = math_utils.complete_square([
                    (old_alpha[t + 1], SIGMA ** 2),
                    eta_mean_var
                ])
            elif t == T:
                alpha_mean, alpha_cov = math_utils.complete_square([
                    (old_alpha[t - 1], SIGMA ** 2),
                    eta_mean_var
                ])
            else:
                alpha_mean, alpha_cov = math_utils.complete_square([
                    (old_alpha[t + 1], SIGMA ** 2),
                    (old_alpha[t - 1], SIGMA ** 2),
                    eta_mean_var
                ])

            alpha[t] = np.random.normal(alpha_mean, alpha_cov)

        # Update eta: stochastic gradient langevin dynamics (SGLD)
        for t in range(T):
            for d in range(D[t]):
                grad_eta = topic_count_by_doc[t][d] - N[t][d] * softmax(eta[t][d])
                eta_prior_grad = -1 / PSI ** 2 * (eta[t][d] - alpha[t])
                eta[t][d, :] += eps / 2. * (grad_eta + eta_prior_grad) + xi

        # Update psi: stochastic gradient langevin dynamics (SGLD)
        old_psi = psi.copy()
        for t in range(T):
            if t == 0:
                psi_prior_grad = 1 / BETA ** 2 * (old_psi[t + 1] - old_psi[t])
            elif t == T:
                psi_prior_grad = 1 / BETA ** 2 * (old_psi[t - 1] - old_psi[t])
            else:
                psi_prior_grad = 1 / BETA ** 2 * (old_psi[t - 1] + old_psi[t + 1] - 2 * old_psi[t])

            grad_psi = topic_count_by_word[t] - np.multiply(topic_count_by_time[t], softmax(old_psi[t]))
            psi[t] += eps / 2. * (grad_psi + psi_prior_grad) + xi

        # Update z: alias table for amortization
        for t in range(T):
            for d in range(D[t]):
                for n in range(N[t][d]):
                    for mh in range(4):
                        k = z[t][d][n]
                        w = self.corpus[t][d][n]
                        topic_count_by_doc[t][d, k] -= 1
                        topic_count_by_word[t, k, w] -= 1
                        topic_count_by_time[t, k] -= 1
                        if mh % 2 == 0:
                            # doc-proposal
                            proposal = None
                            accept = np.exp(psi[t, proposal, w] - psi[t, k, w])
                        else:
                            # word-proposal
                            if nsamples_alias[t, w] >= K:
                                self.update_alias(t, w)
                                self.nsamples_alias[t, w] = 0
                            proposal = self.sample_alias(t, w)
                            self.nsamples_alias[t, w] += 1
                            accept = np.exp(eta[t][d, proposal] - eta[t][d, k])
                        accept = min(accept, 1)
                        new_k = proposal if math_utils.coin(accept) else k
                        z[t][d][n] = new_k
                        topic_count_by_doc[t][d, new_k] += 1
                        topic_count_by_word[t, new_k, w] += 1
                        topic_count_by_time[t, new_k] += 1

        return alpha, eta, psi, z


