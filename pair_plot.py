import matplotlib.pyplot as plt
from feature_scaling import *
import seaborn as sns
import numpy as np
import pandas as pd

def pair_plot(file):
    sns.set(style="ticks", color_codes=True)

    points = pd.read_csv(file).head(50)
    points = points.dropna()
    house_column = 'Hogwarts House'
    
    no_int = points.select_dtypes(include=['object'])
    only_int = points.select_dtypes(exclude=['object'])
    min_c = only_int.apply(mean_normalization)

    houses = points.loc[:, house_column]
    only_int.loc[:, house_column] = houses.loc[:]
    sns.pairplot(points, hue=house_column, size=2,)
    plt.show()
