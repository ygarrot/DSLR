import matplotlib.pyplot as plt
from feature_scaling import *
import numpy as np
import pandas as pd

def scatter(file):
    points = pd.read_csv(file).head(50)
    no_int = points.select_dtypes(include=['object'])
    only_int = points.select_dtypes(exclude=['object'])
    min_c = only_int.apply(mean_normalization)

    # min_c.plot.hist(density = True, bins =5, stacked=True, histtype='bar', alpha=0.5, fill=True)
    # only_int.plot.scatter(x=min_c.columns, y=min_c.columns)
    ax1 = only_int.plot(kind='scatter', x='Index', y='Divination', color='r')    
    ax2 = only_int.plot(kind='scatter', x='Index', y='Astronomy', color='g', ax=ax1)    
    # ax3 = df.plot(kind='scatter', x='e', y='f', color='b', ax=ax1)

    plt.show()
