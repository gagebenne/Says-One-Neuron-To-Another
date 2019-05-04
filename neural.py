import numpy as np
from numpy import genfromtxt

def sigmoid(x):
    return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
    return np.exp(-x) * (1.0 - np.exp(-x))**2

class NeuralNetwork:
    def __init__(self, x, y, nodes):
        self.nodes      = nodes
        self.x          = x
        self.y          = y
        self.hidden     = [np.random.rand(self.nodes, self.x.shape[1]) for _ in range(0, 2)]
        self.weights    = [np.random.rand(self.nodes, self.x.shape[0])] + [np.random.rand(self.nodes, self.nodes)] + [np.random.rand(self.y.shape[0], self.nodes)]
        self.output     = np.zeros(self.y.shape)
        self.samples     = self.y.shape[0]

    def feedforward(self):
        self.hidden[0] = sigmoid(np.dot(self.weights[0], self.x))
        # print(self.weights[0].shape)
        self.hidden[1] = sigmoid(np.dot(self.weights[1], self.hidden[0]))
        # print(self.weights[1].shape)
        self.output = np.dot(self.weights[2], self.hidden[1])
        # print(self.weights[2].shape)
        print("=== feedfoward complete ===")

    def backprop(self):
        d_weights    = [None for _ in range(0, 3)]
        d_weights[2] = np.dot(self.weights[2].T, (self.output - self.y) * sigmoid_derivative(self.output)) * sigmoid_derivative(self.hidden[1])
        # print(np.dot(self.output, d_weights[2].T).shape)
        d_weights[1] = np.dot(self.weights[1].T, d_weights[2]) * sigmoid_derivative(self.hidden[0])
        # print(np.dot(self.hidden[1], d_weights[1].T).shape)
        d_weights[0] = np.dot(self.weights[0].T, d_weights[1]) * sigmoid_derivative(self.x)
        # print(np.dot(self.hidden[0], d_weights[0].T).shape)

        self.weights[2] += np.dot(self.output, d_weights[2].T)
        self.weights[1] += np.dot(self.hidden[1], d_weights[1].T)
        self.weights[0] += np.dot(self.hidden[0], d_weights[0].T)

        print("=== backprop complete ===")

if __name__ == "__main__":
    nn = NeuralNetwork(X, Y, 4)

    for i in range(1500):
        nn.feedforward()
        nn.backprop()
