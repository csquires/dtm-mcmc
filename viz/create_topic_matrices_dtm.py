from viz import viz_map_dynamics
import numpy as np
from utils import read_dats
import json

RES_FOLDER = 'results/blei/twitter1/lda-seq/'
K = 10
T = 10

D = read_dats.read_D('dat_files/twitter/twitter-seq.dat')
ixs2docs = read_dats.read_doc_ids('dat_files/twitter/doc_id.dat', D)
sites2ixs = read_dats.sites2ixs(ixs2docs)

etas = []
a = np.loadtxt(RES_FOLDER + 'gam.dat').reshape([sum(D), K])
a = a / a.sum(axis=1)[:, np.newaxis]
c = 0
for t, d in enumerate(D):
    etas.append(a[c:c+d, :])
    c += d

titles = [
    'Aug. 2013',
    'Sep. 2013',
    'Oct. 2013',
    'Nov. 2013',
    'Dec. 2013',
    'Jan. 2014',
    'Feb. 2014',
    'March 2014',
    'April 2014',
    'May 2014',
]
colors = json.load(open('viz/dtm_colors.json'))

times = [6, 7, 8, 9]
topics = [3]
plt = viz_map_dynamics.create_topics_matrix(
    etas,
    sites2ixs,
    time_names=titles,
    topics=topics,
    times=times,
    filename='figs/topic_matrices_dtm_f3.png',
    colors=colors
)
