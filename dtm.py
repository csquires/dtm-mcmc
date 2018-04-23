import numpy as np
from utils import math_utils


class DynamicTopicModel:
    def __init__(self, corpus, K, V):
        """

        :param corpus: pre-tokenized list of timeslices, each being a list of documents
        :param K: number of topics
        :param V: vocabulary size
        """
        self.corpus = corpus

        T = len(corpus)
        D = [len(timeslice) for timeslice in corpus]
        N = [[len(doc) for doc in timeslice] for timeslice in corpus]

        # sizes
        self.K = K
        self.V = V
        self.T = T
        self.D = D
        self.N = N

        # parameters
        self.alpha = np.zeros([T, K])  # overall topic mixture at time t
        self.eta = [np.zeros([D[t], K]) for t in range(T)]  # d-th document topic mixture at time t
        self.phi = np.zeros([T, K, V])  # k-th topic at time t (as mixture of V words)
        self.z = [[np.zeros(N[t][d], dtype=int) for d in range(D[t])] for t in range(T)]

        # count matrices
        self.topic_count_by_doc = [np.zeros([D[t], K], dtype=int) for t in range(T)]
        self.topic_count_by_time = np.zeros([T, K], dtype=int)
        self.topic_count_by_word = np.zeros([T, K, V], dtype=int)

        # hyperparameters
        self.BURN_IN = 100
        self.THIN_RATE = 10
        self.ALPHA_VAR = 1  # SIGMA in paper
        self.ETA_VAR = 1  # PSI in paper
        self.PHI_VAR = 1  # BETA in paper
        self.EPS_A = 10  # step-size
        self.EPS_B = 1  # step-size
        self.EPS_C = 1.5  # step-size

        # alias tables
        self.nsamples_alias = np.zeros([T, V], dtype=int)
        self.alias_tables = [[None for w in range(V)] for t in range(T)]
        self.prob_tables = [[None for w in range(V)] for t in range(T)]

    def check_counts(self):
        total_num_words = sum(len(doc) for timeslice in self.corpus for doc in timeslice)
        a = self.topic_count_by_time.sum() == total_num_words
        print(a)

    def initialize(self, verbose=False):
        for t in range(self.T):
            for d in range(self.D[t]):
                for n in range(self.N[t][d]):
                    w = self.corpus[t][d][n]
                    k = np.random.randint(0, self.K)
                    self.z[t][d][n] = k
                    self.topic_count_by_doc[t][d, k] += 1
                    self.topic_count_by_time[t, k] += 1
                    self.topic_count_by_word[t, k, w] += 1
                    self.eta[t][d, k] = None
        if verbose:
            print("Done initializing count matrices")

        init_beta = 1
        for t in range(self.T):
            for w in range(self.V):
                for k in range(self.K):
                    num = self.topic_count_by_word[0, k, w] + init_beta
                    denom = self.topic_count_by_time[0, k] + self.V * init_beta
                    self.phi[t, k, w] = num / denom

        if verbose:
            print("Done initializing topic mixtures")

        self.check_counts()

    def sample(self, n_samples, verbose=False):
        samples = []
        for i in range(self.BURN_IN + self.THIN_RATE * n_samples):
            if verbose:
                print('Iteration %d' % i)
            alpha, eta, phi, z = self.gibbs_iter(i)
            if i > self.BURN_IN and (i - self.BURN_IN) % self.THIN_RATE == 0:
                samples.append({
                    'alpha': alpha,
                    'eta': eta,
                    'phi': phi,
                    'z': z
                })
        return samples

    def update_alias(self, t, w):
        prob_table, alias_table = math_utils.get_alias_table(math_utils.softmax_(self.phi[t, :, w]))
        self.alias_tables[t][w] = alias_table
        self.prob_tables[t][w] = prob_table

    def sample_alias(self, t, w):
        if self.alias_tables[t][w] is None:
            self.update_alias(t, w)
        alias_table = self.alias_tables[t][w]
        prob_table = self.prob_tables[t][w]
        self.nsamples_alias[t][w] += 1
        if self.nsamples_alias[t][w] == self.K:
            self.alias_tables[t][w] = None
        return math_utils.sample_alias_table(alias_table, prob_table)

    def gibbs_iter(self, i):
        # sizes
        K = self.K
        T = self.T
        D = self.D
        N = self.N

        # hyperparameters
        ALPHA_VAR = self.ALPHA_VAR
        ETA_VAR = self.ETA_VAR
        PHI_VAR = self.PHI_VAR
        EPS_A = self.EPS_A
        EPS_B = self.EPS_B
        EPS_C = self.EPS_C

        # parameters
        alpha = self.alpha
        eta = self.eta
        phi = self.phi
        z = self.z

        # count matrices
        topic_count_by_doc = self.topic_count_by_doc
        topic_count_by_word = self.topic_count_by_word
        topic_count_by_time = self.topic_count_by_time

        eps = EPS_A * (EPS_B + i) ** EPS_C
        xi = np.random.normal(0, eps ** 2, K)

        # Update alpha: closed form sample from normal
        old_alpha = alpha.copy()
        for t in range(T):
            eta_mean = eta[t].mean(axis=0)  # in R^K
            eta_mean_var = (eta_mean, ETA_VAR ** 2 / D[t])
            if t == 0:
                alpha_mean, alpha_cov = math_utils.complete_square([
                    (old_alpha[t + 1], ALPHA_VAR ** 2),
                    eta_mean_var
                ])
            elif t == T-1:
                alpha_mean, alpha_cov = math_utils.complete_square([
                    (old_alpha[t - 1], ALPHA_VAR ** 2),
                    eta_mean_var
                ])
            else:
                alpha_mean, alpha_cov = math_utils.complete_square([
                    (old_alpha[t + 1], ALPHA_VAR ** 2),
                    (old_alpha[t - 1], ALPHA_VAR ** 2),
                    eta_mean_var
                ])

            alpha[t] = np.random.multivariate_normal(alpha_mean, alpha_cov)

        # Update eta: stochastic gradient langevin dynamics (SGLD)
        for t in range(T):
            for d in range(D[t]):
                grad_eta = topic_count_by_doc[t][d] - N[t][d] * math_utils.softmax_(eta[t][d])
                eta_prior_grad = -1 / ETA_VAR ** 2 * (eta[t][d] - alpha[t])
                eta[t][d, :] += eps / 2. * (grad_eta + eta_prior_grad) + xi

        # Update phi: stochastic gradient langevin dynamics (SGLD)
        old_phi = phi.copy()
        for t in range(T):
            if t == 0:
                phi_prior_grad = 1 / PHI_VAR ** 2 * (old_phi[t + 1] - old_phi[t])
            elif t == T-1:
                phi_prior_grad = 1 / PHI_VAR ** 2 * (old_phi[t - 1] - old_phi[t])
            else:
                phi_prior_grad = 1 / PHI_VAR ** 2 * (old_phi[t - 1] + old_phi[t + 1] - 2 * old_phi[t])

            grad_phi = topic_count_by_word[t] - (topic_count_by_time[t][:, np.newaxis] * math_utils.softmax_(old_phi[t]))
            phi[t] += eps / 2. * (grad_phi + phi_prior_grad) + xi[:, np.newaxis]

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
                            # TODO: I'm still not convinced this is right
                            proposal = z[t][d][np.random.randint(0, N[t][d])]
                            accept = np.exp(phi[t, proposal, w] - phi[t, k, w])
                        else:
                            # word-proposal
                            proposal = self.sample_alias(t, w)
                            accept = np.exp(eta[t][d, proposal] - eta[t][d, k])
                        accept = min(accept, 1)
                        new_k = proposal if math_utils.coin(accept) else k
                        z[t][d][n] = new_k
                        topic_count_by_doc[t][d, new_k] += 1
                        topic_count_by_word[t, new_k, w] += 1
                        topic_count_by_time[t, new_k] += 1

        return alpha, eta, phi, z


