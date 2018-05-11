import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import yaml
import pickle
from palettable import colorbrewer
from utils import strings, math_utils
import itertools as itr


FOLDER = 'results/trials2/'


def plot_proportions(alphas):
    """
    Given alpha (T, K), plot the overall popularity of each topic over time from left to right
    """
    T, K = alphas.shape
    colors = colorbrewer.get_map('Set3', map_type='qualitative', number=min(K, 12)).mpl_colors
    colors = list(itr.islice(itr.cycle(colors), K))
    W = .5
    H = 5
    PW = .2
    TOTAL_W = T * (W + PW)
    plt.figure(figsize=(TOTAL_W, H))
    ax = plt.gca()
    plt.xlim(0, TOTAL_W)
    plt.ylim(0, H)
    plt.axis('off')

    top_by_time = [np.cumsum(alphas[t]) for t in range(T)]
    for t in range(T):
        x = t * (W + PW)
        for k in range(K):
            prop = alphas[t][k]
            y = top_by_time[t][k] - prop
            color = colors[k]
            r = mpatches.Rectangle(xy=(x, y*H), width=W, height=prop*H, edgecolor='k', facecolor=color)
            ax.add_patch(r)

            if t != 0:
                left_y = top_by_time[t-1][k] - alphas[t-1][k]/2
                right_y = y + prop/2
                ax.arrow(x - PW, left_y*H, dx=PW*3/4, dy=(right_y - left_y)*H, head_length=PW/4, head_width=.1)

    return plt


def plot_topics(
        phis,
        int2word,
        time_names=None,
        times=None,
        topics=None,
        ntop=10,
        W=2.2,
        H=.8,
        PW=.2,
        PH=.1,
        arrows=True,
        colors=None,
        max_word_length=20,
        filename=None
):
    """
    Given matrix phi (T, K, V) of log-probabilities of words for each topic at each time,
    plot the top n most likely words in each topic, with time leading from left to right
    """

    T, K, V = phis.shape
    if times is None:
        times = range(T)
    if topics is None:
        topics = range(K)

    TOTAL_W = len(times) * (W + PW)
    TOTAL_H = len(topics) * (H + PH)
    FONTSIZE = 5

    plt.figure(figsize=(TOTAL_W, TOTAL_H))
    plt.tight_layout()
    plt.xlim(0, TOTAL_W)
    plt.ylim(0, TOTAL_H)
    plt.axis('off')
    ax = plt.gca()

    for i, k in enumerate(reversed(topics)):
        y = (i + .5) * (H + PH)
        plt.text(-.2, y, str(k+1), ha='center', fontsize=FONTSIZE+3)

    topn_time_topic = np.argsort(phis)[:, :, ::-1][:, :, :ntop]
    for j, t in enumerate(times):
        print('t =', t)
        print(len(times))
        x = j * (W + PW)
        print(j)
        if time_names is not None:
            plt.text(x + W/2, TOTAL_H+.01, time_names[t], ha='center', va='bottom', fontsize=FONTSIZE+3)

        for i, k in enumerate(reversed(topics)):
            # print('k =', k)
            y = i * (H + PH)

            # draw rectangle
            col = colors[t][k] if colors is not None else 'k'
            r = mpatches.Rectangle(xy=(x, y), width=W, height=H+.02, edgecolor=col, facecolor='none', linewidth=2)
            ax.add_patch(r)

            # draw arrow
            if j != len(times) - 1 and arrows:
                ax.arrow(x + W, y + H/2, dx=PW*3/4, dy=0, head_width=.1, head_length=PW*1/4)

            # draw words
            for m, w in enumerate(topn_time_topic[t, k]):
                word = int2word[w]
                word = word if len(word) < max_word_length else word[:max_word_length - 3] + '...'
                plt.text(x + W / 2, y + H - m * H / ntop, word, ha='center', va='top', fontsize=FONTSIZE)

    if filename is not None:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    return plt


