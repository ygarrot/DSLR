import pandas as pd
import numpy as np
import csv

from Ft_logistic_regression import Ft_logistic_regression

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def get_value(thetas, elem):
    #test = [1] + elem[2:]
    #print (test)
    #print (thetas)
    #return (np.dot(thetas, test))
    ret = thetas[0]
    for i in range(len(thetas) - 1):
        if not np.isnan(elem[i + 2]):
            ret += thetas[i + 1] * elem[i + 2]
    return (sigmoid(ret))

def predict():
    raw_data = pd.read_csv('data/dataset_test.csv',delimiter=',')
    data = raw_data.drop(columns = ['First Name', 'Last Name', 'Birthday', 'Best Hand'])
    #data = data.drop(columns = ['Arithmancy', 'Defense Against the Dark Arts', 'History of Magic', 'Potions', 'Care of Magical Creatures' , 'Charms', 'Flying' ])
    data = data.drop(columns = ['Astronomy', 'Arithmancy', 'Care of Magical Creatures'])
    #hogwarts = data.drop(columns = ['Astronomy', 'Herbology', 'Divination', 'Muggle Studies','Ancient Runes', 'Transfiguration'])
    #hogwarts = hogwarts.drop(columns = ['Arithmancy', 'Defense Against the Dark Arts', 'History of Magic', 'Potions', 'Care of Magical Creatures' , 'Charms', 'Flying' ])
    hogwarts = data.drop(columns = [ 'Herbology', 'Divination', 'Muggle Studies','Ancient Runes', 'Transfiguration'])
    hogwarts = hogwarts.drop(columns = ['Defense Against the Dark Arts', 'History of Magic', 'Potions', 'Charms', 'Flying' ])
    houses = [None] * 4
    thetas = np.genfromtxt('thetas.csv', delimiter=',')
    #print(thetas[0][0])
    for elem in hogwarts['Index']:
        #print(data.loc[elem])
        test = data.loc[elem].tolist()

        #test[2] = (test[2] - min(data['Astronomy'])) / (max(data['Astronomy']) - min(data['Astronomy']))
        #test[3] = (test[3] - min(data['Herbology'])) / (max(data['Herbology']) - min(data['Herbology']))
        #test[4] = (test[4] - min(data['Divination'])) / (max(data['Divination']) - min(data['Divination']))
        #test[5] = (test[5] - min(data['Muggle Studies'])) / (max(data['Muggle Studies']) - min(data['Muggle Studies']))
        #test[6] = (test[6] - min(data['Ancient Runes'])) / (max(data['Ancient Runes']) - min(data['Ancient Runes']))
        #test[7] = (test[7] - min(data['Transfiguration'])) / (max(data['Transfiguration']) - min(data['Transfiguration']))


        #test[2] = (test[2] - min(data['Arithmancy'])) / (max(data['Arithmancy']) - min(data['Arithmancy']))
        #test[3] = (test[3] - min(data['Astronomy'])) / (max(data['Astronomy']) - min(data['Astronomy']))
        test[2] = (test[2] - min(data['Herbology'])) / (max(data['Herbology']) - min(data['Herbology']))
        test[3] = (test[3] - min(data['Defense Against the Dark Arts'])) / (max(data['Defense Against the Dark Arts']) - min(data['Defense Against the Dark Arts']))
        test[4] = (test[4] - min(data['Divination'])) / (max(data['Divination']) - min(data['Divination']))
        test[5] = (test[5] - min(data['Muggle Studies'])) / (max(data['Muggle Studies']) - min(data['Muggle Studies']))
        test[6] = (test[6] - min(data['Ancient Runes'])) / (max(data['Ancient Runes']) - min(data['Ancient Runes']))
        test[7] = (test[7] - min(data['History of Magic'])) / (max(data['History of Magic']) - min(data['History of Magic']))
        test[8] = (test[8] - min(data['Transfiguration'])) / (max(data['Transfiguration']) - min(data['Transfiguration']))
        test[9] = (test[9] - min(data['Potions'])) / (max(data['Potions']) - min(data['Potions']))
        #test[12] = (test[12] - min(data['Care of Magical Creatures'])) / (max(data['Care of Magical Creatures']) - min(data['Care of Magical Creatures']))
        test[10] = (test[10] - min(data['Charms'])) / (max(data['Charms']) - min(data['Charms']))
        test[11] = (test[11] - min(data['Flying'])) / (max(data['Flying']) - min(data['Flying']))
        #print(test)
        #print(test)
        houses[0] = get_value(thetas[0], test)
        houses[1] = get_value(thetas[1], test)
        houses[2] = get_value(thetas[2], test)
        houses[3] = get_value(thetas[3], test)
        #print(houses)
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
