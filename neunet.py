import numpy as np
from scipy.special import expit

def sigmoid(x):
    return expit(x)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.randn(self.input.shape[1], 10) * np.sqrt(1 / self.input.shape[1])
        self.weights2 = np.random.randn(10, 1) * np.sqrt(1 / 10)
        self.bias1 = np.zeros((1, 10))
        self.bias2 = np.zeros((1, 1))
        self.y = y
        self.output = np.zeros(y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1) + self.bias1)
        self.output = sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, (2 * (self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(
            self.input.T,
            (np.dot(2 * (self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T)
             * sigmoid_derivative(self.layer1))
        )

        self.weights1 += d_weights1
        self.weights2 += d_weights2
        self.bias1 += np.sum((2 * (self.y - self.output) * sigmoid_derivative(self.output)) * sigmoid_derivative(self.layer1), axis=0)
        self.bias2 += np.sum(2 * (self.y - self.output) * sigmoid_derivative(self.output), axis=0)


# TRAINING DATA: 
# Sample input array (customer information)
x = np.array([[100000, 6, 10000, 1, 200000, 50000, 750, 35],
              [50000, 4, 5000, 0, 100000, 30000, 650, 30],
              [200000, 5, 20000, 1, 300000, 80000, 700, 45],
              [150000, 5.5, 15000, 1, 250000, 60000, 720, 40]])

# Sample target (label) array indicating loan status (0: no default, 1: default)
y = np.array([[1],
              [0],
              [1],
              [1]])

# Normalize the input data
x_normalized = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

# Create an instance of NeuralNetwork
nn = NeuralNetwork(x_normalized, y)

# Training loop
epochs = 100000  # Number of training iterations
for i in range(epochs):
    nn.feedforward()
    nn.backprop()

# After training, you can use the trained network to make predictions
nn.feedforward()
print("Predicted Output:")
print(nn.output)

# Save the weights and biases
np.save("weights1.npy", nn.weights1)
np.save("weights2.npy", nn.weights2)
np.save("bias1.npy", nn.bias1)
np.save("bias2.npy", nn.bias2)