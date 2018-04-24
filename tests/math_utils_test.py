from utils import math_utils
import numpy as np


def test_complete_square():
    mu1 = np.array([1, 1])
    mu2 = np.array([2, 0])
    sigma1 = 1
    sigma2 = 5
    mean, cov = math_utils.complete_square([(mu1, sigma1), (mu2, sigma2)])
    correct_cov = (1/sigma1 + 1/sigma2) ** -1
    correct_mean = correct_cov * (mu1/sigma1 + mu2/sigma2)
    print("mean correct:", mean == correct_mean)


mu1 = np.array([1, 1])
cov = np.eye(2)
ys = np.random.multivariate_normal(mu1, cov, size=100)
mu_prior = np.array([0, 0])
y_mean_covs = [(y, 1) for y in ys]
post_mean, post_cov = math_utils.complete_square([*y_mean_covs, (mu_prior, 1)])



# ps = math_utils.softmax_(np.random.random(size=1000))
# prob_table, alias_table = math_utils.get_alias_table(ps)
# samples = math_utils.sample_alias_table(alias_table, prob_table, size=400000)
# ps_emp = np.bincount(samples)/len(samples)
# print(np.allclose(ps, ps_emp, atol=1e-2))

