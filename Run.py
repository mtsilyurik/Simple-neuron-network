import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron
from SimpleNeuron import Neuron

# x1 = 2, x2 = 3
Xi = np.array([2, 3])

# w1 = 0, w2 = 1
Wi = np.array([0, 1])

basis = [Neuron(Wi, b).feedforward(Xi) for b in np.arange(-4, 4, 1)]

plt.plot(basis)
plt.xlabel("Basis")
plt.ylabel("sigmoid(x)")
plt.grid(True)
plt.show()

