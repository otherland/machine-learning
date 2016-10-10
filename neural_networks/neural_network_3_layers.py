import numpy as np

def nonlin(x,derivative=False):
    if derivative:
        return x*(1-x)
    else:
        return 1 / (1 + np.exp(-x))


# predict exam grade given past performance data:
# hours sleep, hours study, exam grade


## ----------------------- Part 1 ---------------------------- ##

# X = (hours sleeping, hours studying), y = Score on test
X = np.array([  [3,5],
                [5,1],
                [2,1],
                [5,5],
                [12,4],
                [10,2] ],
                dtype=float)

y = np.array([  [75],
                [82],
                [45],
                [90],
                [95],
                [93] ],
                dtype=float)

## ----------------------- Part 2 ---------------------------- ##
# normalize/scale inputs
# xnorm = X / max(X)
# ynorm = y /max(y)

X = X / np.amax(X, axis=0)

# we specify 100 as the max test score
# because test input may not contain a score of 100
y /= 100


## ----------------------- Part 3 ---------------------------- ##
"""
deterministic random
give random numbers that are generated the same starting point (seed=1) so that we'll get the same sequence of generated of generated numbers every program run
"""
np.random.seed(1)

# randomly initialize our weights
syn0 = np.random.randn(2,3)
syn1 = np.random.randn(3,1)


## ----------------------- Part 4 ---------------------------- ##
#
# TRAINING
#
for j in range(60000):

    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # How much did we miss the target value?
    l2_error = y - l2

    if j % 10000 == 0:
        print("Error:", np.mean(np.abs(l2_error)))

    """
    Multiply error rate by result of sigmoid function
    function used to get derivative of the output prediction from layer_3
    this will give us a delta which we'll use to reduce the error rate of our predictions when we update our weights every iteration
    """
    l2_delta = l2_error * nonlin(l2, derivative=True)

    """
    Backpropogation:
    How much did layer2 contribute to error in layer3?
    1) Multiply layer3 delta by weights2 transposed
    2) Get layer2_delta by multiplying layer2_error
    by result of sigmoid function. Result is used to get derivative of layer2
    """

    l1_error = l2_delta.dot(syn1.T)

    # In what direction is the target l1?
    # Were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1, derivative=True)

    """
    Gradient descent:
    Update synapse weights using deltas to reduce error rate each iteration
    """
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

print("Output after training")
print("Actual output:\n", y)
print("Predicted output:\n", l2)