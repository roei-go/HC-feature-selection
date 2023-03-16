import matplotlib.pyplot as plt
import numpy as np
def plot_hist(ax,arrays,bins,xlabel,title,log_scale=False,labels=None):
    for i in range(len(arrays)):
        if labels is not None:
            label = labels[i]
        else:
            label = None
        ax.hist(arrays[i], bins=bins, log=log_scale, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('count')
    ax.set_title(title)
    if labels is not None:
        ax.legend()


def bar_plot_scores(scores, bar_labels, xticklabels, xlabel, ylabel, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xticks(np.arange(scores.shape[1]))
    width = 1/(2*scores.shape[0])
    bar_distances = np.arange(scores.shape[0]) - scores.shape[0]/2
    for i in range(scores.shape[0]):
        ax.bar(x=np.arange(scores.shape[1]) - bar_distances[i]*width, height=scores[i,:], width=width, label=bar_labels[i])
    ax.set_xlabel(xlabel)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='lower center')
    plt.show()