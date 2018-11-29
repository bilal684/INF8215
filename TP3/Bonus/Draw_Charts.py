import numpy as np
import pandas as pd

#labels is a Series or a

import numpy as np
import matplotlib.pyplot as plt
def bins_labels(bins, **kwargs):
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
    plt.xlim(bins[1], bins[-1])



def Draw_Charts(labels,graph_title):
    no_classes=len(np.unique(labels))
    bins = range(0,no_classes+1)
    (n,bins1,patches1)=plt.hist(labels, bins,
             align='mid',weights=np.ones(len(labels)) / len(labels))
    bins_labels(bins, fontsize=20)
    plt.xlabel("Class")
    plt.ylabel("Ratio")
    i=0
    for a,b in zip(bins1,patches1):
        plt.text(b.get_x()+0.5, n[i]+0.01, r'{0:.4f}'.format(n[i]), ha='center', va='bottom', fontsize=7)
        i=i+1
    plt.title(graph_title)
    plt.show()