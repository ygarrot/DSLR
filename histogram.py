import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pandas as pd

def histogram(file, get_head=False):
    sns.set(style="ticks", color_codes=True)

    house_column = 'Hogwarts House'
    hist_col = 'Arithmancy'

    points = pd.read_csv(file).dropna()
    only_int = pd.DataFrame(points.select_dtypes(exclude=['object']))

    if (get_head is True):
        for col in only_int:
            cur_col = pd.DataFrame([points[house_column], only_int[col]]).T
            sns.catplot(x=house_column, y=col, kind='bar', data=cur_col)
            plt.show()
            return

    cur_col = pd.DataFrame([points[house_column], only_int[hist_col]).T
    sns.catplot(x=house_column, y=hist_col, kind='bar', data=cur_col)
    plt.show()

if __name__ == '__main__':
    if (sys.argv[1]):
        histogram(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == "-a")
