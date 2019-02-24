import matplotlib.pyplot as plt
from feature_scaling import *
import numpy as np
import pandas as pd

def pair_plot(file):
    points = pd.read_csv(file).head(50)
    no_int = points.select_dtypes(include=['object'])
    only_int = points.select_dtypes(exclude=['object'])
    min_c = only_int.apply(mean_normalization)
    pd.scatter_matrix(only_int)
    plt.show()

