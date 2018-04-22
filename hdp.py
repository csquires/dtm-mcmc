import gensim
from gensim.models import hdpmodel
from gensim.corpora import Dictionary
from gensim.test.utils import common_dictionary, common_corpus, common_texts

stopwords = ['and']


def doc2rep(doc):
    return [w for w in doc.lower().split() if w not in stopwords]


corpus = [
    'spain spain spain',
    'france paris',
    'spain and food'
]
corpus = [doc2rep(doc) for doc in corpus]
dictionary = Dictionary(corpus)
corpus = [dictionary.doc2bow(doc) for doc in corpus]
hdp = hdpmodel.HdpModel(corpus, dictionary)