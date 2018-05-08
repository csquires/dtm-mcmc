import numpy as np
import os
from utils import sys_utils

K = 10
T = 12
# FOLDER_ESC = '/Users/chandlersquires/Dropbox\ \(MIT\)/School/_Spring2018/6882/Project/code'
FOLDER = '/Users/chandlersquires/Dropbox (MIT)/School/_Spring2018/6882/Project/code'

GET_BLEI_RESULTS = True
if GET_BLEI_RESULTS:
    os.chdir('/Users/chandlersquires/Desktop/dtm-master/dtm/')
    os.system('pwd')
    for i in range(1, 100):
        outfolder = '%s/results/blei/sim_corpora/corpus%d' % (FOLDER, i)
        sys_utils.ensure_dir(outfolder + '/')
        print(outfolder)
        cmd = './main --ntopics=%d' % K
        cmd += ' --mode=fit --rng_seed=0 --initialize_lda=true'
        cmd += ' --corpus_prefix="%s/dat_files/sim_corpora/corpus%d/corpus%d"' % (FOLDER, i, i)
        cmd += ' --outname="%s"' % outfolder
        cmd += ' --top_chain_var=0.005 --alpha=0.01'
        cmd += ' --lda_sequence_min_iter=6 --lda_sequence_max_iter=20 --lda_max_em_iter=10'
        os.system(cmd)
    os.chdir(FOLDER)


# for i in range(100):
#     for k in range(K):
#         topic_fn = 'topic-%03d-var-e-log-prob.dat' % k
#         log_word_dist = np.loadtxt(topic_fn).reshape([-1, T])
#         word_dist = np.exp(log_word_dist)




