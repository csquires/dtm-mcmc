import matplotlib.pyplot as plt
import numpy as np
import json

topics_over_time = json.load(open('topics_over_time.json'))
t = 0
n_topics = len(topics_over_time[t])
n_words = 10
fig, axes = plt.subplot(n_topics, 1)

for ax, topic in zip(axes, topics_over_time[t]):
    ax.imshow(topic, cmap='seismic')
    ax.set_xticks([])
    ax.set_yticks(range(n_words))
    ax.set_yticklabels(topic, size=14)



