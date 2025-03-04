import matplotlib.pyplot as plt
import numpy as np

# Load and process data
Pin_1 = np.loadtxt("./Pin_transient_non_linear_1e-2.dat") * 0.00075
Pin_1 = Pin_1[-100:]  # Last 100 points
Pin_2 = np.loadtxt("./Pin_transient_non_linear_1e-3.dat") * 0.00075
Pin_2 = Pin_2[-1000:]  # Last 1000 points

# Create normalized x-axes (from 0 to 1) for both datasets
x1 = np.linspace(0, 1, len(Pin_1))
x2 = np.linspace(0, 1, len(Pin_2))

# Plot using the normalized x-axes
plt.figure(figsize=(10, 6))
plt.plot(x1, Pin_1, label="1e-2")
plt.plot(x2, Pin_2, label="1e=3")

# Calculate and plot difference (resampling one to match the other)
# Uncomment if needed:
# Pin_2_resampled = np.interp(x1, x2, Pin_2)
# plt.plot(x1, Pin_1 - Pin_2_resampled, label="Difference")

plt.grid()
plt.legend()
plt.xlabel("Normalized Time (0-1)")
plt.ylabel("Value")
plt.title("Comparison of Old and New Code Results")
plt.show()
