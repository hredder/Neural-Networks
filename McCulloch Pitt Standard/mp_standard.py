import numpy as np

# Class for a McCulloch Pitt Network with standard input and bipolar output

class MPStandardNeuron:

    # Creates the perceptron with perceptron_size different input values
    def __init__(self, weights, bias):

        # Initialize all weights to propagate forward to 1
        self.inputWeights = weights

        # Initialize bias to 0
        self.bias = bias


    # Classifies an input vector to a binary class (either true or false)
    def evaluate(self, inputVals):

        inner_prod = np.dot(self.inputWeights, inputVals) + self.bias
        if (inner_prod >= 0):
            return 1
        else:
            return -1

