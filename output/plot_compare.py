import matplotlib.pyplot as plt
import numpy as np

Pin_1 = np.loadtxt("./Pin_transient_non_linear_1e-2.dat") * 0.00075
Pin_1 = Pin_1[-100:]
Pin_2 = np.loadtxt("./Pin_transient_non_linear_1e-3.dat") * 0.00075
Pin_2 = Pin_2[-1000:]
plt.plot(Pin_1, label="New Code")
plt.plot(Pin_2, label="Old Code")
# plt.plot(Pin_new - Pin_old, label="Difference")
plt.grid()
plt.legend()
plt.show()
