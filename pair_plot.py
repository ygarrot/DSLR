import matplotlib.pyplot as plt
from feature_scaling import *
import seaborn as sns
import numpy as np
import pandas as pd

# def slice_name(array)
#     line =
#     for index in array
#         index

def pair_plot(file):
    sns.set(style="ticks", color_codes=True)

    points = pd.read_csv(file)
    points = points.dropna()
    house_column = 'Hogwarts House'

    only_int = points.select_dtypes(exclude=['object'])
    min_c = only_int.apply(mean_normalization)
    # only_int.apply(remove, index=[0])

    houses = points.loc[:, house_column]
    only_int.loc[:, house_column] = houses.loc[:]
    sns.pairplot(points, hue=house_column)
    plt.show()
