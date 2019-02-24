import matplotlib.pyplot as plt
from feature_scaling import rescaling
import numpy as np
import pandas as pd

def histogram(file):
    points = pd.read_csv(file).head(50)
    no_int = points.select_dtypes(include=['object'])
    only_int = points.select_dtypes(exclude=['object'])
    # print(no_int)
    house_col = 'Hogwarts House'
    houses = no_int.groupby(house_col)[house_col].first()
    # houses = houses[house_col]
    # houses = houses.first()
    all_houses = [ house for house in houses ]
    print(all_houses)
    return
    no_int = points.select_dtypes(include=['object'])
    # points.filter()
    # points = points.sort_values('Hogwarts House', ascending=False)
    min_c = only_int.apply(rescaling)

    # min_c.plot.hist(density=True, histtype='bar')
    min_c.plot.hist( density=True, histtype='bar', stacked=True, alpha=0.5)
    # min_c.plot.hist( histtype='step', stacked=True, fill=False)
    # min_c.plot.hist(alpha=0.5,histtype='bar')

    # only_int['Divination'].plot.hist(stacked=True )
    # plt.show()
    return

