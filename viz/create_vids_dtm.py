from viz import viz_map_dynamics
import numpy as np
from utils import read_dats, sys_utils

RES_FOLDER = 'results/blei/twitter1/lda-seq/'
K = 10
T = 10

D = read_dats.read_D('dat_files/twitter/twitter-seq.dat')
ixs2docs = read_dats.read_doc_ids('dat_files/twitter/doc_id.dat', D)
sites2ixs = read_dats.sites2ixs(ixs2docs)

# 33 sites missing doc at some time

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


times = [
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
for t, (y, m) in zip(range(10), times):
    print(t)
    for topic in range(10):
        sys_utils.ensure_dir('figs/dtm/topic%d' % topic)
        filename = 'figs/dtm/topic%d/topic%d_%d_%02d.png' % (topic, topic, y, m)
        viz_map_dynamics.create_topics_snapshot(etas, sites2ixs, topic, t, title=titles[t], filename=filename)

# for topic in range(10):
#     viz_map_dynamics.create_topics_animation(etas, sites2ixs, 'vids/dtm/topic%d.mp4' % topic, topic, titles)