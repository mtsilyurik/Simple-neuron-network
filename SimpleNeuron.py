import random
import numpy as np

def sigmoid(x) -> float:
    return 1 / (1 + np.exp(-x))

class Neuron:
    # activation function f(x) = 1 / (1+e^(-x))
    def __init__(self, w, b):
        self.w = w
        self.b = b

    # Sum
    def feedforward(self, x) -> float:
        s = np.dot(x, self.w.T) + self.b
        return sigmoid(s)


# # Entries
# # x1 = 2, x2 = 3
# # Xi = np.array([random.randint(-5, 5) for i in range(10)])
# Xi = np.array([2, 3])
# print(Xi)
# # Weights
# # w1 = 0, w2 = 1
# # Wi = np.array([random.choice([0, 1]) for i in range(Xi.size)])
# Wi = np.array([0, 1])
# print(Wi)
# # b = 4
# # basis = -4
#
# for b in range(-4, 4, 1):
#     neuron = Neuron(Wi, b)
#     print(f"b = {b} --- sigmoid(S) = {neuron.feedforward(Xi)}")
