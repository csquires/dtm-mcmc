from viz import viz_map_dynamics
from gensim.models import LdaModel
from utils import read_dats, sys_utils
import numpy as np

T = 10
K = 10
V = 10124

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

D = read_dats.read_D('dat_files/twitter/twitter-seq.dat')
ixs2sites = read_dats.read_doc_ids('dat_files/twitter/doc_id.dat', D)
sites2ixs = read_dats.sites2ixs(ixs2sites)

etas, phis = read_dats.lda2mats('results/lda/twitter/', 'dat_files/twitter/twitter-mult.dat', times, D, K, V)

# idea: color each topic at time t > 1 the same as the most similar topic at time t=1
# todo: order sites consistently by similarity

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
for t, (y, m) in zip(range(10), times):
    print(t)
    for topic in range(10):
        sys_utils.ensure_dir('figs/lda/topic%d' % topic)
        filename = 'figs/lda/topic%d/topic%d_%d_%02d.png' % (topic, topic, y, m)
        viz_map_dynamics.create_topics_snapshot(etas, sites2ixs, topic, t, title=time_names[t], filename=filename)

# for topic in range(10):
#     viz_map_dynamics.create_topics_animation(etas, sites2ixs, 'vids/topic%d.mp4' % topic, topic, time_names)


