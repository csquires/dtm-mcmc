from viz import viz_map_dynamics
from utils import read_dats
import json

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

time_names = [
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
colors = json.load(open('viz/lda_colors.json'))

topics = None
times = None
plt = viz_map_dynamics.create_topics_matrix(
    etas,
    sites2ixs,
    time_names=time_names,
    topics=topics,
    times=times,
    filename='figs/topic_matrices_lda.png',
    colors=colors
)
