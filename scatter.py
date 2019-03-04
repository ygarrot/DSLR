import matplotlib.pyplot as plt
from feature_scaling import *
import seaborn as sns
import numpy as np
import pandas as pd

def scatter(file):
    sns.set(style="ticks", color_codes=True)

    points = pd.read_csv(file)
    only_int = points.select_dtypes(exclude=['object'])
    sns.scatterplot(data=only_int,x='Astronomy', y='Defense Against the Dark Arts')

    plt.show()
