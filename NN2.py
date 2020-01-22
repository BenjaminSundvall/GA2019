import numpy as np
import matplotlib.pyplot as plt


class FirstNN(object):
    def __init__(self):
        # Define HyperParameters
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        # Weights
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)

    def forward(self, X):
        # Propagate inputs through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    '''
    testInput = np.arange(-6, 6, 0.01)
    plt.plot(testInput, sigmoid(testInput), linewidth=2)
    plt.grid(1)
    '''


'''
NN = FirstNN()
yHat = NN.forward(X)
'''
