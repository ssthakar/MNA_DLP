import matplotlib.pyplot as plt
import numpy as np

Pin_new = np.loadtxt("./Pin_transient.dat")
Pin_old = np.loadtxt("./Pin_transient_old.dat")

# Sample every nth point for scatter plot
n = 5  # Adjust this value to control point density
x = np.arange(len(Pin_new))
x_scattered = x[::n]
Pin_new_scattered = Pin_new[::n]

# Scatter plot with reduced points, but line plot with all points
plt.scatter(x_scattered, Pin_new_scattered, label="New Code", alpha=0.6, s=20)
plt.plot(x, Pin_old, label="Old Code")
plt.grid()
plt.legend()
plt.show()
