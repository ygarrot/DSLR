import sys
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def scatter(file, get_head=False):
    sns.set(style="ticks", color_codes=True)

    points = pd.read_csv(file)
    if (get_head is True):
        points = points.head()
    only_int = points.select_dtypes(exclude=['object'])
    sns.scatterplot(data=only_int,x='Astronomy', y='Defense Against the Dark Arts')
    plt.show()


if __name__ == '__main__':
    if (sys.argv[1]):
        pair_plot(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == "-h")
