import numpy as np

class Perceptron:

    # Creates the perceptron with perceptron_size different input values
    def __init__(self, perceptron_size, learning_rate):

        # Note weights[0] is bias corresponding to +1 input 
        self.weights = np.zeros(perceptron_size+1)

        # Set constant learning rate value
        self.constant_learning_rate_val = learning_rate

        # Set time step which equals how many vectors have been trained
        self.time_step = 0

        # The learning rate of the perceptron
        self.learning_rate_param = self.constant_learning_rate

    # Updates the weights of the neural network to be 
    def update(self, input_vec, predicted_class):

        # If predicted class is equal to the evaluated input vector then don't modify weights
        # otherwise modify the weight vector

        input_vec = np.insert(input_vec, 0, 1)
        evaluation = self.evaluate(input_vec)
        print(evaluation)
        self.weights += self.learning_rate_param(self.time_step)* input_vec * (predicted_class - evaluation)
        self.time_step += 1

    # Classifies an input vector to a binary class (either true or false)
    def evaluate(self, input_vec):

        inner_prod = np.dot(self.weights, input_vec)
        if (inner_prod >= 0):
            return 1
        else:
            return -1

    # Returns the constant learning rate value
    def constant_learning_rate(self, n):
        return self.constant_learning_rate_val