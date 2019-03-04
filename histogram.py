import matplotlib.pyplot as plt
import seaborn as sns
from feature_scaling import *
import numpy as np
import pandas as pd

def histogram(file):
    sns.set(style="ticks", color_codes=True)

    house_column = 'Hogwarts House'
    hist_col = 'Care of Magical Creatures'

    points = pd.read_csv(file).dropna()

    only_int = points.select_dtypes(exclude=['object'])
    only_int.apply(mean_normalization)
    CMC = only_int[hist_col]
    CMC = pd.DataFrame([points[house_column],CMC]).T
    sns.countplot(x=house_column, data=CMC)
    plt.show()
