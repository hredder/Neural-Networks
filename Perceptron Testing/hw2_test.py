from perceptron import Perceptron
import numpy as np

perc = Perceptron(2, 0.5)
x_1 = np.array([0,0]) #in c_1
x_2 = np.array([0,1]) #in c_1
x_3 = np.array([1,0]) #in c_2
x_4 = np.array([1,1]) #in c_2

print(perc.weights)
perc.update(input_vec=x_1, predicted_class=1)
print(perc.weights)
perc.update(input_vec=x_2, predicted_class=-1)
print(perc.weights)
perc.update(input_vec=x_3, predicted_class=-1)
print(perc.weights)
perc.update(input_vec=x_4, predicted_class=1)
print(perc.weights)

perc.update(input_vec=x_1, predicted_class=1)
print(perc.weights)
perc.update(input_vec=x_2, predicted_class=-1)
print(perc.weights)
perc.update(input_vec=x_3, predicted_class=-1)
print(perc.weights)
perc.update(input_vec=x_4, predicted_class=1)
print(perc.weights)

perc.update(input_vec=x_1, predicted_class=1)
print(perc.weights)
perc.update(input_vec=x_2, predicted_class=-1)
print(perc.weights)
perc.update(input_vec=x_3, predicted_class=-1)
print(perc.weights)
perc.update(input_vec=x_4, predicted_class=1)
print(perc.weights)