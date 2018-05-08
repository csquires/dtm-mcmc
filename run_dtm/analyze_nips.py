from dtm import DynamicTopicModel
from utils import sys_utils
import yaml
import io

FOLDER = 'results/nips/'
sys_utils.ensure_dir(FOLDER)
FILENAME = FOLDER + 'samples2.p'


corpus = []


for i in range(1990, 2003):
    timeslice = []
    with io.open('data/nips_data/%d.txt' % i) as f:
        for line in f.readlines():
            timeslice.append([int(w) for w in line.split()])
    corpus.append(timeslice)

int2word = {}
with io.open('data/nips_data/vocabulary_file.txt') as f:
    for i, line in enumerate(f.readlines()):
        int2word[i] = line.strip()
yaml.dump(int2word, open(FOLDER + 'int2word.yaml', 'w'))
V = len(int2word)


K = 10
d = DynamicTopicModel(corpus, K, V)
d.BURN_IN = 1
d.THIN_RATE = 1
d.initialize(verbose=True)
d.sample(10, verbose=True)
d.save(FILENAME)
