import numpy as np
from scipy.special import expit

def sigmoid(x):
    return expit(x)

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

class NeuralNetwork:
    def __init__(self, x):
        self.input = x
        self.weights1 = np.load("weights1.npy")
        self.weights2 = np.load("weights2.npy")
        self.bias1 = np.load("bias1.npy")
        self.bias2 = np.load("bias2.npy")
        self.output = np.zeros((x.shape[0], 1))

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1) + self.bias1)
        self.output = sigmoid(np.dot(self.layer1, self.weights2) + self.bias2)

# New data without expected results
x_new = np.array([[80000, 5, 8000, 1, 150000, 40000, 700, 36],
                  [120000, 6, 12000, 0, 250000, 70000, 680, 42]])

# Normalize the new input data using the same mean and standard deviation as before
x_new_normalized = (x_new - np.mean(x_new, axis=0)) / np.std(x_new, axis=0)

# Create an instance of NeuralNetwork
nn = NeuralNetwork(x_new_normalized)

# Run the feedforward step to make predictions
nn.feedforward()

# Print the predicted outputs
print("Predicted Output:")
print(nn.output)