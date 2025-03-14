import matplotlib.pyplot as plt
import numpy as np

# Load and scale data
P1 = np.loadtxt("./tracked_1.dat") * 0.00075  # Ground truth
P2 = np.loadtxt("./tracked_2.dat") * 0.00075  # Unoptimized params
P3 = np.loadtxt("./tracked_3.dat") * 0.00075  # Optimized params

# First figure: Two separate subplots for inlet and outlet
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Subplot 1: Inlet pressure (first column)
ax1.plot(P1[:, 0], label="Ground truth", linewidth=2)
ax1.plot(P2[:, 0], label="Unoptimized params", linestyle="--")
# ax1.plot(P3[:, 0], label="Optimized params", linestyle="-.")
ax1.set_ylabel("Inlet Pressure")
ax1.legend()
ax1.grid(True)
ax1.set_title("Inlet Pressure Comparison")

# Subplot 2: Outlet pressure (second column)
ax2.plot(P1[:, 1], label="Ground truth", linewidth=2)
ax2.plot(P2[:, 1], label="Unoptimized params", linestyle="--")
# ax2.plot(P3[:, 1], label="Optimized params", linestyle="-.")
ax2.set_xlabel("Time Steps")
ax2.set_ylabel("Outlet Pressure")
ax2.legend()
ax2.grid(True)
ax2.set_title("Outlet Pressure Comparison")

# Adjust layout
plt.tight_layout()

# Save first figure
fig1.savefig("pressure_comparison.svg", format="svg")
fig1.savefig("pressure_comparison.png", dpi=300)

# Second figure: Combined inlet and outlet in single plot
fig2, ax = plt.subplots(figsize=(10, 6))

# Plot inlet pressures
ax.plot(P1[:, 0] - P1[:, 1], label="Pressure Difference", linewidth=2, color="blue")
# ax.plot(P1[:, 1], label="Outlet (Ground truth)", linestyle="--", color="red")

# Plot outlet pressures
# ax.plot(P2[:, 0], label="Inlet (Unoptmized)", linewidth=2, color="lightblue")
# ax.plot(P2[:, 1], label="Outlet (Unoptimized)", linestyle="--", color="salmon")

ax.set_xlabel("Time Steps")
ax.set_ylabel("Pressure")
ax.legend()
ax.grid(True)
# ax.set_title("Combined Inlet and Outlet Pressure Comparison")

# Adjust layout
plt.tight_layout()

# Save second figure
fig2.savefig("combined_pressure_comparison.svg", format="svg")
fig2.savefig("combined_pressure_comparison.png", dpi=300)

# Show plots
plt.show()
