import numpy as np
import matplotlib.pyplot as plt

class Ft_logistic_regression():

    """
    This class handle LogisticRegression for a given dataset, it detects how many features there is in the dataset
    and uppon calling gradient descent, set the variable raw_thetas to the result of n epochs of gradient descent on starting raw_thetas.
    All values in X are normalized so regression is faster.

    when we access j, we access a feature, when we access i, we access a sample.

    for example, accessing X[i,j] means accessing the jth feature of the ith sample.
    we only access data in y with i and data in thetas with j for clarity

    attributes:

    m is the number of samples in the dataset
    n is the number of features in the dataset

    raw_data is the data sent by user. it's never overwritten.
    raw_thetas are the starting thetas sent by user. it's overwritten at the end of a gradient_descent call


    X is the normalized data array of size m * n + 1. The first column of X is filled with 1s for quick hypothesis computation
    y is the values vector of size m * 1.

    thetas are the normalized thetas vector of size n + 1. They're updated at each epochs of the gradient_descent function
    """

    def __init__(self, thetas, data, epochs = 1000, learning_rate = 0.001):
        self.cost = []
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.raw_thetas = thetas
        self.raw_data = data
        self.m = self.raw_data.shape[0];
        self.n = self.raw_data.shape[1] - 1;
        self.__get_scaled_data()
        try:
            self.__get_scaled_thetas()
        except:
            print('error in raw_thetas format, setting thetas to 0')
            self.thetas = np.zeros(self.n)
            self.raw_thetas = np.zeros(self.n)

    def gradient_descent(self):
        for n in range (0, self.epochs):
            self.thetas = self.__gradient_descent_epoch()
            #self.cost.append(self.get_cost())
        self.raw_thetas = np.empty(len(self.thetas))
        for j in range(self.n):
            self.raw_thetas[j] = self.thetas[j] / (max(self.raw_data[:, j]) - min(self.raw_data[:, j]))

    def get_cost(self):
        cost = 0;
        for i in range (1, self.m):
            cost += self.y[i] * np.log(self.__predict(i)) + (1 - self.y[i]) * np.log(1 - self.__predict(i))
        cost /= float(self.m)
        return -cost

    def show(self):
        plt.subplot(1, 2, 1)
        plt.plot(self.raw_data[:, 0], self.raw_data[:, 1], 'r.')
        print(max(self.raw_data[:, 0]))
        t0 = np.mean(self.y) - (np.mean(self.raw_data[:, 0]) * self.thetas[1])
        plt.plot([0, max(self.raw_data[:, 0])], [self.raw_thetas[0], self.raw_thetas[0] + self.raw_thetas[1] * max(self.raw_data[:, 0])])
        plt.ylabel('y')
        plt.xlabel('x')
        plt.subplot(1, 2, 2)
        plt.plot(self.cost)
        plt.ylabel('cost')
        plt.xlabel('epochs')
        plt.tight_layout()
        plt.show()

    # Adds a column filled with 1 (So Theta0 * x0 = Theta0) and apply MinMax normalization to the raw data
    def __get_scaled_data(self):
        self.X = np.empty(shape=(self.m, self.n)) # create the data matrix of size m * n
        self.y = np.empty(shape=(self.m, 1))
        self.y = self.raw_data[:, self.raw_data.shape[1] - 1] # copy y values of rawdata to y vector
        # assign raw data to X matrix
        for j in range(0, self.n):
            self.X[:, j] = self.raw_data[:, j]
        # normalize the raw data stored in X matrix using mean max normalization
        for j in range(self.n):
            self.X[:, j] = (self.X[:, j] - min(self.raw_data[:, j])) / (max(self.raw_data[:, j]) - min(self.raw_data[:, j]))

    def __get_scaled_thetas(self):
        self.thetas = np.empty(self.n)
        for j in range(0, self.n):
            self.thetas[j] = self.raw_thetas[j] * (max(self.raw_data[:, j]) - min(self.raw_data[:, j]))

    def __gradient_descent_epoch(self):
        new_thetas = np.zeros(self.n)
        for i in range(self.m):
            delta = self.__predict(i) - self.y[i]
            for j in range(self.n):
                if not np.isnan(self.X[i,j]):
                    new_thetas[j] += delta * self.X[i, j]
        for j in range(self.n):
            new_thetas[j] = self.thetas[j] - (self.learning_rate / float(self.m)) * new_thetas[j]
        return new_thetas

    def __predict(self, i):
        h = 0
        for j in range(self.n):
            if not (np.isnan(self.X[i, j])):
                h += self.thetas[j] * self.X[i, j]
        h = self.__sigmoid(h)
        return h

    def __sigmoid(self, h):
        return 1 / (1 + np.exp(-h))
