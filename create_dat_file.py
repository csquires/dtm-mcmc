import io
from collections import Counter
from utils import sys_utils
DAT_FOLDER = 'dat_files/sim_corpora/'


def create_dat_files(corpus, title):
    T = len(corpus)

    folder = DAT_FOLDER + title + '/'
    sys_utils.ensure_dir(folder)
    seq_filename = folder + '%s-seq.dat' % title
    mult_filename = folder + '%s-mult.dat' % title
    with io.open(seq_filename, 'w') as seq_file, io.open(mult_filename, 'w') as mult_file:
        seq_file.write('%d\n' % T)
        for t in range(T):
            d = len(corpus[t])
            seq_file.write('%d\n' % d)
            for d in range(d):
                c = Counter()
                c.update(corpus[t][d])
                mult_file.write(str(len(c)))
                for w, count in c.items():
                    mult_file.write(' %d:%d' % (w, count))
                mult_file.write('\n')


if __name__ == '__main__':
    import json
    import pickle

    for i in range(100):
        corpus = pickle.load(open('data/sim_corpora/corpus%d.p' % i, 'rb'))['ws']
        create_dat_files(corpus, 'corpus%d' % i)


