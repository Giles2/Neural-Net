import numpy as np
from scipy.special import expit

def sigmoid(x):
    return expit(x)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.randn(self.input.shape[1], 4) * np.sqrt(1 / self.input.shape[1])
        self.weights2 = np.random.randn(4, 1) * np.sqrt(1 / 4)
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.layer1, self.weights2))

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(
            self.input.T,
            (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T)
             * sigmoid_derivative(self.layer1))
        )

        self.weights1 += d_weights1
        self.weights2 += d_weights2

# Sample input array (customer information)
x = np.array([[30, 50000],
              [25, 35000],
              [40, 75000],
              [35, 60000],
              [45, 90000]])

# Sample target (label) array indicating loan default (1) or non-default (0)
y = np.array([[0],
              [1],
              [0],
              [0],
              [1]])

# Create an instance of NeuralNetwork
nn = NeuralNetwork(x, y)

# Training loop
epochs = 100000  # Number of training iterations
for i in range(epochs):
    nn.feedforward()
    nn.backprop()

# After training, you can use the trained network to make predictions
nn.feedforward()
print("Predicted Output:")
print(nn.output)

    