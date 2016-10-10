
# predict exam grade given past performance data:
# hours sleep, hours study, exam grade


## ----------------------- Part 1 ---------------------------- ##
import numpy as np

# X = (hours sleeping, hours studying), y = Score on test
X = np.array(([3,5], [5,1], [10,2]), dtype=float)
y = np.array(([75], [82], [93]), dtype=float)


## ----------------------- Part 2 ---------------------------- ##
# normalize/scale inputs
# xnorm = X / max(X)
# ynorm = y /max(y)

X = X / np.amax(X, axis=0)

# we specify 100 as the max test score
# because test input may not contain a score of 100
y /= 100


## ----------------------- Part 3 ---------------------------- ##

class Neural_Network(object):
    def __init__(self):
        # Define HyperParameters
        self.input_layer_size = 2
        self.output_layer_size = 1
        self.hidden_layer_size = 3

        # Initialize Weights (Parameters)
        # Create matrices with random values
        self.W1 = np.random.rand(self.input_layer_size, \
                                 self.hidden_layer_size)
        self.W2 = np.random.randn(self.hidden_layer_size, \
                                  self.output_layer_size)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, X):
        """
        Pass multiple inputs through the network at once using matrices
        """
        # np.dot does matrix multiplication
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        # estimate of y
        yHat = self.sigmoid(self.z3)
        return yHat

    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W1 and W2
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

if __name__ == '__main__':
    NN = Neural_Network()
    yHat = NN.forward(X)
    print(yHat)