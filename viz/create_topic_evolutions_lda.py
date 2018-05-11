from viz import viz_topics
import numpy as np
from palettable import colorbrewer, cartocolors
import json
from utils import read_dats
from scipy.spatial.distance import cosine
from scipy.stats import entropy

RES_FOLDER = 'results/blei/twitter1/lda-seq/'
K = 10
T = 10
N_TOP = 10
word2int = json.load(open('dat_files/twitter/word2int.json'))
int2word = dict(map(reversed, word2int.items()))
V = len(int2word)
D = read_dats.read_D('dat_files/twitter/twitter-seq.dat')


TIMES = [
    (2013, 8),
    (2013, 9),
    (2013, 10),
    (2013, 11),
    (2013, 12),
    (2014, 1),
    (2014, 2),
    (2014, 3),
    (2014, 4),
    (2014, 5),
]

time_names = [
    'August 2013',
    'September 2013',
    'October 2013',
    'November 2013',
    'December 2013',
    'January 2014',
    'February 2014',
    'March 2014',
    'April 2014',
    'May 2014',
]


etas, phis = read_dats.lda2mats('results/lda/twitter/', 'dat_files/twitter/twitter-mult.dat', TIMES, D, K, V)
first_time_colors = cartocolors.qualitative.Bold_10.mpl_colors
colors = [first_time_colors]
for t in range(T):
    time_t_colors = []
    for k in range(K):
        topic = phis[t, k]
        cos_dists = [entropy(phis[0][k_], topic) for k_ in range(K)]
        closest = np.argmax(cos_dists)
        time_t_colors.append(first_time_colors[closest])
    colors.append(time_t_colors)
json.dump(colors, open('viz/lda_colors.json', 'w'), indent=2)


plt = viz_topics.plot_topics(
    phis,
    int2word,
    time_names=time_names,
    arrows=False,
    colors=colors,
    ntop=8,
    W=1.5,
    H=1,
    filename='figs/topic_evolutions_lda.png'
)
