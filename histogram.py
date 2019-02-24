import matplotlib.pyplot as plt
from feature_scaling import rescaling
import numpy as np
import pandas as pd

def histogram(file):
    points = pd.read_csv(file).head(50)
    # points = points.dropna()
    no_int = points.select_dtypes(include=['object'])
    # only_int.sort_values
    only_int = points.select_dtypes(exclude=['object'])
    points = points.sort_values('Hogwarts House', ascending=False)
    min_c = only_int.apply(rescaling)
    # a_heights, a_bins = np.histogram(only_int['Divination'])
    min_c.plot.hist(stacked=True ,alpha=0.5)
    # only_int['Divination'].plot.hist(stacked=True )
    plt.show()
    return

    n, bins, patches = plt.hist(only_int.T, 10, histtype='bar', stacked=True, label =None)#, fill=False, orientation='vertical')
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    # plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])
    plt.grid(True)
    plt.show()
    
   # file = 
