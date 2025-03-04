import matplotlib.pyplot as plt
import numpy as np

# Load and process data
Pin_1 = np.loadtxt("./Pin_transient_non_linear_1e-2.dat") * 0.00075
Pin_1 = Pin_1[-101:]  # Last 100 points
Pin_2 = np.loadtxt("./Pin_transient_non_linear_1e-3.dat") * 0.00075
Pin_2 = Pin_2[-1001:]  # Last 1000 points
Pin_3 = np.loadtxt("./Pin_transient_old.dat") * 0.00075
Pin_3 = Pin_2[-1001:]  # Last 1000 points


# Plot Pin_1 normally
plt.plot(Pin_1, label="1e-2")

# Plot every 10th point from Pin_2 to match the length of Pin_1
# plt.plot(Pin_2[::10], label="1e-3")
plt.plot(Pin_3[::10], label="old_implementation")

# If you want to plot the difference:
# Pin_2_decimated = Pin_2[::10]
# plt.plot(Pin_1 - Pin_2_decimated, label="Difference")

plt.grid()
plt.legend()
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Comparison of Old and New Code Results")
plt.show()
