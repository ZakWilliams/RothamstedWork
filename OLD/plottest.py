import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(5, 50, 1000)

gvals = np.exp(-((x-2)/4)**2)
evals = np.exp(2*(x-5))

y = gvals * evals

plt.plot(x, y)
plt.show()