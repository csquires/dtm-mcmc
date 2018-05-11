from viz import viz_topics
import numpy as np
import json
from palettable import cartocolors


RES_FOLDER = 'results/blei/twitter1/lda-seq/'
K = 10
T = 10
word2int = json.load(open('dat_files/twitter/word2int.json'))
int2word = dict(map(reversed, word2int.items()))
V = len(int2word)
n_top = 10

phis = np.zeros([T, K, V])
for k in range(K):
    print(k)
    prefix = 'results/blei/twitter1/lda-seq/topic-%03d' % k
    topic_probs = np.loadtxt(prefix + '-var-e-log-prob.dat').reshape([-1, T])  # word probability per topic
    phis[:, k, :] = topic_probs.T

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
colors = cartocolors.qualitative.Bold_10.mpl_colors
colors = [colors for i in range(T)]
json.dump(colors, open('viz/dtm_colors.json', 'w'), indent=2)

# times = [6, 7, 8, 9]
# topics = [6, 7]

times = [6, 7, 8, 9]
topics = [3]
plt = viz_topics.plot_topics(
    phis,
    int2word,
    times=times,
    topics=topics,
    time_names=time_names,
    ntop=20,
    W=1.5,
    H=4,
    colors=colors,
    filename='figs/topic_evolutions_dtm_f3.png'
)


