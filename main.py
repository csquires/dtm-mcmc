from dtm import DynamicTopicModel
import itertools as itr
from datetime import datetime
import json
from utils import strings
import pandas as pd


def decode_topics(samples, int2word, top_n=10):
    phis = [s['phi'] for s in samples]
    decoded = []
    for t in range(len(phis)):
        decoded.append([[int2word[w] for w in phi[:top_n]] for phi in phis])
    return decoded


corpus = []
dates = list(itr.product(range(2013, 2014), range(1, 13)))


for i in range(len(dates) - 1):
    d1 = datetime(*dates[i], 1)
    d2 = datetime(*dates[i + 1], 1)
    docs = json.load(open('data/docs_%s_%s.json' % (d1, d2)))
    timeslice = [doc['title'] for doc in docs]
    corpus.append(timeslice)

corpus_tokenized, int2word, word_counts = strings.tokenize_corpus(corpus)
json.dump(dict(word_counts), open('words.json', 'w'), indent=2)
V = len(int2word)

K = 10
d = DynamicTopicModel(corpus_tokenized, K, V)
d.initialize(verbose=True)
samples = d.sample(100, verbose=True)
df = pd.DataFrame(samples)
df.to_csv('samples/1.txt')

