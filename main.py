import numpy as np
import pandas as pd
from Ft_array import *
import argparse
from pandas.api.types import is_string_dtype

def main():
    parser = argparse.ArgumentParser(description='process linear regression')
    parser.add_argument(type=str, dest="dataset",
                             help="describe dataset")
    
    opt = parser.parse_args()
    # points = pd.read_csv(opt.dataset).head()
    points = pd.read_csv(opt.dataset).head()

    only_int = points.select_dtypes(exclude=['object'])
    mean = only_int.mean()
    std = only_int.std()
    # print(only_int)
    # for column in only_int:
        # my_min = ft_min(only_int[column])
    # print(my_min)
    count = only_int.apply(ft_count)
    std = only_int.apply(ft_std)
    mean = only_int.apply(ft_mean)
    median = only_int.apply(ft_median)
    first_quar = only_int.apply(ft_first_quar)
    third_quar = only_int.apply(ft_third_quar)
    median = only_int.apply(ft_median)
    min_c = only_int.apply(ft_min)
    max_c = only_int.apply(ft_max)
    # mean = only_int.mean()
    # mean = only_int.mean()
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
    # print(pd.DataFrame(std).T.to_string(col_space=2, header=False))
    # print(pd.DataFrame(min_c).T.to_string(col_space=2, header=False))
    # print(pd.DataFrame(max_c).T.to_string(justify='left',col_space=2, header=False))
    # print(pd.DataFrame(mean).T.to_string(header=False))

    # print(points.select_dtypes(exclude=['object']).describe())
    # print(pd.DataFrame(points, columns=['First Name']))
    # print(is_string_dtype(pd.DataFrame(points, columns=['First Name'])))

    # points = np.genfromtxt(opt.dataset, delimiter=",")
    # print(points)

if __name__ == '__main__':
    main()
