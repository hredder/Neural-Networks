from perceptron import Perceptron
import matplotlib.pyplot as plt
import numpy as np

perc = Perceptron(2, 0.5)

points_1_x = np.random.normal(loc = 0, scale=1.0, size=100)
points_1_y = np.random.normal(loc = 0, scale=1.0, size=100)

points_2_x = np.random.normal(loc = 0, scale=1.0, size=100) + 4
points_2_y = np.random.normal(loc = 0, scale=1.0, size=100) + 4

plt.scatter(points_1_x, points_1_y, 3, None, 'x')
plt.scatter(points_2_x, points_2_y, 3, None, 'x')
epochs = 10

for j in range(epochs):
    for i in range(0, 100):
        perc.update(input_vec=np.array([points_1_x[i], points_1_y[i]]), predicted_class=True)
        perc.update(input_vec=np.array([points_2_x[i], points_2_y[i]]), predicted_class=False)


final_weights = perc.weights
x_1 = -5
x_2 = 5
y_1 = (-x_1 * final_weights[1] - final_weights[0]) / final_weights[2]
y_2 = (-x_2 * final_weights[1] - final_weights[0]) / final_weights[2]

plt.axline((x_1, y_1), (x_2, y_2))
plt.show()


