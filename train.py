import pandas as pd
import numpy as np
from Ft_logistic_regression import Ft_logistic_regression

def train():
    raw_data = pd.read_csv('data/dataset_train.csv',delimiter=',')
    data = raw_data.drop(columns = ['First Name', 'Last Name', 'Birthday', 'Best Hand', 'Arithmancy', 'Defense Against the Dark Arts', 'History of Magic', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying' ])
    hogwarts = {
            'Gryffindor':data.copy(),
            'Slytherin':data.copy(),
            'Hufflepuff':data.copy(),
            'Ravenclaw':data.copy()
            }
    #    for elem in data['Index']:
#        hogwarts[data.loc[elem, 'Hogwarts House']] = np.vstack((hogwarts[data.loc[elem, 'Hogwarts House']], data.loc[elem]))

    hogwarts_thetas = {
            'Gryffindor':[],
            'Slytherin':[],
            'Hufflepuff':[],
            'Ravenclaw':[]
            }
    for key in hogwarts:
        for elem in hogwarts[key]['Index']:
            hogwarts[key].loc[elem, 'Hogwarts House'] = 1 if hogwarts[key].loc[elem, 'Hogwarts House'] == key else 0
        hogwarts[key] = np.array(hogwarts[key])
        hogwarts[key] = np.array(hogwarts[key][:,1:])
        hogwarts[key] = np.hstack((hogwarts[key][:,1:], hogwarts[key][:,:1]))
        lr = Ft_logistic_regression(epochs = 2000, learning_rate = 0.4, data = hogwarts[key], thetas = [0,0,0,0,0,0])
        print('Training ' + key + '...')
        lr.gradient_descent()
        hogwarts_thetas[key] = lr.raw_thetas
        print ('Thetas values for ' + key + ' : ')
        print(hogwarts_thetas[key])
        print('Done training ' + key + ' !\n\n')

    thetas= np.array([hogwarts_thetas['Gryffindor'],hogwarts_thetas['Slytherin'],hogwarts_thetas['Hufflepuff'],hogwarts_thetas['Ravenclaw']])
    print(thetas)
    print(thetas.shape)
    np.savetxt('thetas.csv', thetas, delimiter=',');

def main():
    train()

if __name__ == '__main__':
    main()
