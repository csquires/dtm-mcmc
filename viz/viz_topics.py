import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml
import pickle
from palettable import colorbrewer
from utils import strings, math_utils
import itertools as itr

plt.ion()

FOLDER = 'results/trials2/'
int2word = yaml.load(open(FOLDER + 'int2word.yaml'))
samples = pickle.load(open(FOLDER + 'samples1.p', 'rb'))
topics = strings.decode_topics(samples.iloc[-1], int2word)
topic_props = np.apply_along_axis(math_utils.softmax_, 1, samples.ix[0]['alpha'])
K = 10
n_top = 10


def plot_proportions(topic_props, times):
    colors = colorbrewer.get_map('Set3', map_type='qualitative', number=min(K, 12)).mpl_colors
    colors = list(itr.islice(itr.cycle(colors), K))
    W = .5
    H = 5
    PW = .2
    TOTAL_W = len(times) * (W + PW)
    plt.figure(figsize=(TOTAL_W, H))
    ax = plt.gca()
    plt.xlim(0, TOTAL_W)
    plt.ylim(0, H)
    plt.axis('off')

    top_by_time = [np.cumsum(topic_props[t]) for t in times]
    for t in times:
        x = t * (W + PW)
        for i in range(len(topic_props[t])):
            prop = topic_props[t][i]
            y = top_by_time[t][i] - prop
            color = colors[i]
            r = mpatches.Rectangle(xy=(x, y*H), width=W, height=prop*H, edgecolor='k', facecolor=color)
            ax.add_patch(r)

            if t != 0:
                left_y = top_by_time[t-1][i] - topic_props[t-1][i]/2
                right_y = y + prop/2
                ax.arrow(x - PW, left_y*H, dx=PW*3/4, dy=(right_y - left_y)*H, head_length=PW/4, head_width=.1)
    plt.show()


def plot_topics(topics, topic_ixs, times):
    W = .5
    H = 1
    PW = .1
    PH = .1
    TOTAL_W = len(times) * (W + PW)
    TOTAL_H = len(topic_ixs) * (H + PH)
    FONTSIZE = 5
    plt.figure(figsize=(TOTAL_W, TOTAL_H))
    plt.tight_layout()
    plt.xlim(0, TOTAL_W)
    plt.ylim(0, TOTAL_H)
    plt.axis('off')
    ax = plt.gca()
    for t in times:
        x = t * (W + PW)
        for i, top in enumerate(topic_ixs):
            y = i * (H + PH)

            # draw rectangle
            r = mpatches.Rectangle(xy=(x, y), width=W, height=H, edgecolor='k', facecolor='none')
            ax.add_patch(r)

            # draw arrow
            if not t == len(times) - 1:
                ax.arrow(x + W, y + H/2, dx=PW*3/4, dy=0, head_width=.1, head_length=PW*1/4)

            # draw words
            for j, w in enumerate(topics[t, top]):
                plt.text(x + W/2, y + H - j * H/n_top, w, ha='center', va='top', fontsize=FONTSIZE)

    plt.show()


# plot_proportions(topic_props, range(4))

topic_ixs = range(10)
times = range(10)
plot_topics(topics, topic_ixs, times)
