import numpy as np
import pandas as pd
from Ft_array import *
import argparse
from pandas.api.types import is_string_dtype

def describe(file):
    # points = pd.read_csv(opt.dataset).head()
    points = pd.read_csv(file).head()

    only_int = points.select_dtypes(exclude=['object'])

    count = only_int.apply(ft_count)
    std = only_int.apply(ft_std)
    mean = only_int.apply(ft_mean)
    median = only_int.apply(ft_median)
    first_quar = only_int.apply(ft_first_quar)
    third_quar = only_int.apply(ft_third_quar)
    median = only_int.apply(ft_median)
    min_c = only_int.apply(ft_min)
    max_c = only_int.apply(ft_max)

    name = ["Count",
            "Mean",
            "Std",
            "Min",
            "25%",
            "50%",
            "75%",
            "Max"]

    print(name)
    print(only_int.describe().to_string())
    print(pd.DataFrame([count, mean, std, min_c, first_quar, median, third_quar, max_c], index=name).to_string(col_space=2))

