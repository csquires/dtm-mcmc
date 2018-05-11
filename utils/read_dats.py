import io
import numpy as np
from gensim.models import LdaModel


def read_D(filename):
    D = []
    with io.open(filename) as f:
        f.readline()
        for line in f.readlines():
            D.append(int(line.strip()))
    return D


def read_corpus(filename, start_line, end_line):
    corpus = []
    with io.open(filename) as f:
        for line in f.readlines()[start_line:end_line]:
            doc = []
            line = line.strip().split()
            for pair in line[1:]:
                wordtok, n = pair.split(':')
                doc.append((int(wordtok), int(n)))
            corpus.append(doc)
    return corpus


def read_doc_ids(filename, D):
    ixs2docs = [[] for d in D]
    with io.open(filename) as f:
        lines = f.readlines()
        c = 0
        for i, d in enumerate(D):
            for line in lines[c:c + d]:
                line = line.strip()
                y, m, lon, lat = line.split('_')
                lon = int(lon[4:])
                lat = int(lat[4:])
                ixs2docs[i].append((lon, lat))
            c += d
    return ixs2docs


def get_all_sites(ixs2sites):
    sites = set()
    for docs in ixs2sites:
        for lon, lat in docs:
            sites.add((lon, lat))
    return sorted(sites)


def sites2ixs(ixs2sites):
    T = len(ixs2sites)
    sites = get_all_sites(ixs2sites)
    sites2ixs = {site: [None] * T for site in sites}
    for t in range(T):
        for d, site in enumerate(ixs2sites[t]):
            sites2ixs[site][t] = d
    return sites2ixs


def lda2mats(lda_folder, corpus_filename, times, D, K, V):
    T = len(times)
    etas = []
    phis = np.zeros([T, K, V])
    c = 0
    for (y, m), d, t in zip(times, D, range(T)):
        print(y, m)
        lda = LdaModel.load(lda_folder + '%d_%02d.lda' % (y, m))
        phis[t] = lda.get_topics()
        corpus = read_corpus(corpus_filename, c, c + d)
        eta = np.zeros([d, K])
        for i, doc in enumerate(corpus):
            print(i)
            doc_topics = lda.get_document_topics(doc)
            for k, p in doc_topics:
                eta[i, k] = p
        etas.append(eta)
        c += d
    return etas, phis
