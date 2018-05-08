from dtm import DynamicTopicModel
import itertools as itr
from datetime import datetime
import json
from utils import strings, sys_utils
import pickle
import yaml
import numpy as np

FOLDER = 'results/trials2/'
sys_utils.ensure_dir(FOLDER)
FILENAME = FOLDER + 'sample3.p'


corpus = []
dates = list(itr.product(range(2013, 2014), range(1, 13)))


for i in range(len(dates) - 1):
    d1 = datetime(*dates[i], 1)
    d2 = datetime(*dates[i + 1], 1)
    docs = json.load(open('data/trials_data/docs_%s_%s.json' % (d1, d2)))
    timeslice = [doc['title'] for doc in docs]
    corpus.append(timeslice)

corpus_tokenized, int2word, word_counts = strings.tokenize_corpus(corpus)
json.dump(corpus_tokenized, open(FOLDER + 'corpus.json', 'w'), indent=2)
json.dump(dict(word_counts), open(FOLDER + 'word_counts.json', 'w'), indent=2)
yaml.dump(int2word, open(FOLDER + 'int2word.yaml', 'w'), indent=2)
V = len(int2word)
# samples = pickle.load(open(FILENAME, 'rb'))
# topic = samples.ix[0]['phi'][0][0]
# print(strings.decode_topic(topic, int2word))
# topics = strings.decode_topics(samples.ix[0], int2word)

# K = 10
# d = DynamicTopicModel(corpus_tokenized, K, V)
# d.BURN_IN = 1
# d.THIN_RATE = 1
# d.initialize(verbose=True)
# d.sample(100, verbose=True)
# d.save(FILENAME)


