import pandas as pd
import numpy as np
from Ft_logistic_regression import Ft_logistic_regression
from threading import Thread

class multi_train(Thread):

    def __init__(self, key, hogwarts, hogwarts_thetas):
        Thread.__init__(self)
        self.key = key
        self.hogwarts = hogwarts
        self.hogwarts_thetas = hogwarts_thetas

    def run(self):
        key = self.key
        hogwarts = self.hogwarts
        hogwarts_thetas = self.hogwarts_thetas
        for elem in hogwarts[key]['Index']:
            hogwarts[key].loc[elem, 'Hogwarts House'] = 1 if hogwarts[key].loc[elem, 'Hogwarts House'] == key else 0
        hogwarts[key] = np.array(hogwarts[key])
        hogwarts[key] = np.array(hogwarts[key][:,1:])
        hogwarts[key] = np.hstack((hogwarts[key][:,1:], hogwarts[key][:,:1]))
        lr = Ft_logistic_regression(epochs = 2000, learning_rate = 20, data = hogwarts[key])
        print('Training ' + key + '...')
        lr.gradient_descent()
        hogwarts_thetas[key] = lr.thetas
        print ('Thetas values for ' + key + ' : ')
        print(hogwarts_thetas[key])
        print('Done training ' + key + ' !\n\n')

def train():
    raw_data = pd.read_csv('data/dataset_train.csv',delimiter=',')
    data = raw_data.drop(columns = ['First Name', 'Last Name', 'Birthday', 'Best Hand'])
    data = data.drop(columns = ['Arithmancy', 'Care of Magical Creatures', 'Astronomy'])
    hogwarts = {
            'Gryffindor':data.copy(),
            'Slytherin':data.copy(),
            'Hufflepuff':data.copy(),
            'Ravenclaw':data.copy()
            }

    hogwarts_thetas = {
            'Gryffindor':[],
            'Slytherin':[],
            'Hufflepuff':[],
            'Ravenclaw':[]
            }
    thread = {}
    for key in hogwarts:
        thread[key] = multi_train(key, hogwarts, hogwarts_thetas)
    for key in thread:
        thread[key].start()
    for key in thread:
        thread[key].join()
    thetas= np.array([hogwarts_thetas['Gryffindor'],hogwarts_thetas['Slytherin'],hogwarts_thetas['Hufflepuff'],hogwarts_thetas['Ravenclaw']])
    np.savetxt('thetas.csv', thetas, delimiter=',')

def main():
    train()

if __name__ == '__main__':
    print("start")
    main()
