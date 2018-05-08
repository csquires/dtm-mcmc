from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from collections import defaultdict, OrderedDict
import operator as op
import re
import numpy as np
import string

STOPWORDS = set(stopwords.words('english'))
trademark = '\u00ae'
SUB_PATTERN = r'[-.\',/\\&`~!@#$%%^&*?"|()=+%s]' % trademark


def process_text(text):
    text = text.lower()
    text = re.sub(SUB_PATTERN, ' ', text)
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)
    words = text.split(' ')
    words = filter(lambda w: w not in STOPWORDS, words)
    words = filter(lambda w: not w.isdigit(), words)

    return list(words)


def process_corpus(corpus):
    return [[process_text(doc) for doc in timeslice] for timeslice in corpus]


def tokenize_corpus(corpus):
    proc_corpus = process_corpus(corpus)
    word_counts = defaultdict(int)
    for w in (word for timeslice in proc_corpus for doc in timeslice for word in doc):
        word_counts[w] += 1
    word_counts = OrderedDict(sorted(word_counts.items(), key=op.itemgetter(1), reverse=True))
    int2word = dict(enumerate(word_counts.keys()))
    word2int = dict(map(reversed, int2word.items()))
    corpus_tokenized = [[[word2int[w] for w in doc] for doc in timeslice] for timeslice in proc_corpus]
    return corpus_tokenized, int2word, word_counts


def decode_topic(topic, int2word, n_top=10):
    top_ixs = np.argsort(topic)[::-1][:n_top]
    return [int2word[ix] for ix in top_ixs]


def decode_topics(sample, int2word, n_top=10):
    dec = lambda t: decode_topic(t, int2word, n_top)
    return np.apply_along_axis(dec, 2, sample['phi'])
