import numpy as np
import math

# Class for an n-dimensional input self-organizing map with 2 output dimensionally arranged neurons
class SelfOrganizingMap:

    # Creates the perceptron with perceptron_size different input values
    def __init__(self, input_count, x_count, y_count):
        self.sigma = 1
        self.input_count = input_count
        self.x_count = x_count
        self.y_count = y_count
        self.weights = []
        self.learning_rate = 0.5
        
        for _ in range(input_count):
            self.weights.append(np.random.standard_normal(size=(x_count, y_count)))


    # Update the weights of a self-organizing map
    def update_weights(self, x):
        for i in self.input_count:
            for x in self.x_count:
                for y in self.y_count:
                    self.weights[i][x][y] += self.learning_rate*self.neighborhood_function()

    
    # Neighboordhood function used for weight updates
    def neighborhood_function(self, i, j, x):
        return math.exp(-1*((i-j)*(i-j)) / (2 * self.sigma * self.sigma))



