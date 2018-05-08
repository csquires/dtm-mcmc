from dtm import DynamicTopicModel, get_ll
import itertools as itr
from datetime import datetime
import json
from utils import strings, sys_utils
import pickle
import yaml
import numpy as np

FOLDER = 'results/sim_corpora/'
sys_utils.ensure_dir(FOLDER)

V = 1000
K = 10

for i in range(8, 100):
    vars = pickle.load(open('data/sim_corpora/corpus%d.p' % i, 'rb'))
    true_ll = get_ll(vars['etas'], vars['phis'], vars['ws'])
    corpus = vars['ws']
    FILENAME = FOLDER + 'corpus%d.p' % i
    print("==== CORPUS %d ====" % i)
    print("True log likelihood:", true_ll)

    d = DynamicTopicModel(corpus, K, V)
    d.BURN_IN = 200
    d.THIN_RATE = 10
    d.initialize(verbose=True)
    d.sample(100, verbose=True)
    d.save(FILENAME)


