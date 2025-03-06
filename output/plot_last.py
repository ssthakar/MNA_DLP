import matplotlib.pyplot as plt
import numpy as np

Pin_new = np.loadtxt("./Pin_transient_non_linear_1e-2.dat") * 0.00075
Pin_new = Pin_new[-1000:]
Pin_old = np.loadtxt("./Pin_transient_old.dat") * 0.00075
Pin_old = Pin_old[-1000:]
plt.plot(Pin_new, label="New Code")
plt.plot(Pin_old, label="Old Code")
# plt.plot(Pin_new - Pin_old, label="Difference")
plt.grid()
plt.legend()
plt.show()
