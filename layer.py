import numpy as np


class Dense:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))
        self.input = None
        self.d_weights = None
        self.d_biases = None

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases

    def backward(self, dout):
        self.d_weights = np.dot(self.input.T, dout)
        self.d_biases = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.weights.T)

    def update(self, lr):
        self.weights -= lr * self.d_weights
        self.biases -= lr * self.d_biases
