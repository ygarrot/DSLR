import pandas as pd
import numpy as np
import csv

from Ft_logistic_regression import Ft_logistic_regression

def get_value(thetas, elem):
    ret = thetas[0]
    for i in range(len(thetas) - 1):
        if not np.isnan(elem[i + 2]):
            ret += thetas[i + 1] * elem[i + 2]
    return (ret)

def predict():
    raw_data = pd.read_csv('data/dataset_test.csv',delimiter=',')
    data = raw_data.drop(columns = ['First Name', 'Last Name', 'Birthday', 'Best Hand', 'Arithmancy', 'Defense Against the Dark Arts', 'History of Magic', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying' ])
    hogwarts = data.drop(columns = ['Astronomy', 'Divination', 'Herbology', 'Muggle Studies','Ancient Runes', 'Transfiguration'])
    houses = [None] * 4
    thetas = np.genfromtxt('thetas.csv', delimiter=',')
    print(thetas[0][0])
    for elem in hogwarts['Index']:
        print(data.loc[elem])
        test = data.loc[elem].tolist()
        print(test)
        houses[0] = get_value(thetas[0], test)
        houses[1] = get_value(thetas[1], test)
        houses[2] = get_value(thetas[2], test)
        houses[3] = get_value(thetas[3], test)
        if (houses[0] > houses[1] and houses[0] > houses[2] and houses[0] > houses[3]):
            hogwarts.loc[elem, 'Hogwarts House'] = 'Gryffindor'
        elif (houses[1] > houses[2] and houses[1] > houses[3]):
            hogwarts.loc[elem, 'Hogwarts House'] = 'Slytherin'
        elif (houses[2] > houses[3]):
            hogwarts.loc[elem, 'Hogwarts House'] = 'Hufflepuff'
        else:
            hogwarts.loc[elem, 'Hogwarts House'] = 'Ravenclaw'
   # print(hogwarts)
   # print(thetas.shape)
    file_content  = hogwarts.to_csv(index=False)
    f = open('houses.csv', 'w+')
    f.write(file_content)
    f.close()
def main():
    predict()

if __name__ == '__main__':
    main()
