import matplotlib.pylab as plt
import numpy as np

x = np.arange(-8, 8, 0.1)

w1, w2, w3 = 0.5, 1.0, 2.0

l1, l2, l3 = f"w1 = {w1}", f"w2 = {w2}", f"w3 = {w3}"

for w, l in [(w1, l1), (w2, l2), (w3, l3)]:
    # sigmoid
    f = 1 / (1 + np.exp(-x * w))

    plt.plot(x, f, label=l)

plt.xlabel("x")
plt.ylabel("Y = f(x)")
plt.legend(loc=4)
plt.show()
