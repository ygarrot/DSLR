import pandas as pd
import numpy as np
from Ft_logistic_regression import Ft_logistic_regression

def train():
    raw_data = pd.read_csv('data/dataset_train.csv',delimiter=',')
    hogwarts = {
            'Gryffindor':np.zeros((0,8)),
            'Slytherin':np.zeros((0,8)),
            'Hufflepuff':np.zeros((0,8)),
            'Ravenclaw':np.zeros((0,8))
            }
    data = raw_data.drop(columns = ['First Name', 'Last Name', 'Birthday', 'Best Hand', 'Arithmancy', 'Defense Against the Dark Arts', 'History of Magic', 'Potions', 'Care of Magical Creatures', 'Charms', 'Flying' ])
    print(type(data))
    print(data.shape)
    for elem in data['Index']:
        hogwarts[data.loc[elem, 'Hogwarts House']] = np.vstack((hogwarts[data.loc[elem, 'Hogwarts House']], data.loc[elem]))

    print(hogwarts['Gryffindor'].shape)
    print(hogwarts['Slytherin'].shape)
    print(hogwarts['Hufflepuff'].shape)
    print(hogwarts['Ravenclaw'].shape)
    train_gry = np.zeros((327, 7));
    train_gry

def main():
    train()

if __name__ == '__main__':
    main()
