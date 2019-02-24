import matplotlib.pyplot as plt
from feature_scaling import *
import numpy as np
import pandas as pd

def histogram(file):
    points = pd.read_csv(file).head(50)
    no_int = points.select_dtypes(include=['object'])
    only_int = points.select_dtypes(exclude=['object'])
    no_int = points.select_dtypes(include=['object'])
    del only_int['Index']
    min_c = only_int.apply(mean_normalization)

    # min_c.plot.hist(density=True, histtype='bar')
    min_c.plot.hist(density = True, bins =5, stacked=True, histtype='bar', alpha=0.5, fill=True)
    # min_c.plot.hist( histtype='step', stacked=True, fill=False)
    # min_c.plot.hist(alpha=0.5,histtype='bar')
    plt.show()

