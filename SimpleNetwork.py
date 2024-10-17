from SimpleNeurom import Neuron
import numpy as np

def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))

class SimpleNetwork:
    def __init__(self):
        weights = np.array([0,1])
        basis = 0
        self.h1 = Neuron(weights, basis)
        self.h2 = Neuron(weights, basis)
        self.o1 = Neuron(weights, basis)

    def feedforward(self, x) -> float:
        out_h1 = self.h1.feedforward(x)
        out_h2 = self.h2.feedforward(x)

        out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))
        return out_o1


network = SimpleNetwork()
x = np.array([2,3])
print(network.feedforward(x))