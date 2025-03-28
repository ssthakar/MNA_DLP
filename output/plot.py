import matplotlib.pyplot as plt
import numpy as np
import os

fig_dir = "figures"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)


tracked_data_new = np.loadtxt("./tracked_data_K.dat")
tracked_data_old = np.loadtxt("./tracked_data_C.dat")

Pin_new = tracked_data_new[:, 0] * 0.00075
Pin_old = tracked_data_old[:, 0] * 0.00075
P1_new = tracked_data_new[:, 1] * 0.00075
P1_old = tracked_data_old[:, 1] * 0.00075
P2_new = tracked_data_new[:, 2] * 0.00075
P2_old = tracked_data_old[:, 2] * 0.00075

P_diff_1_new = Pin_new - P1_new
P_diff_2_new = Pin_new - P2_new

P_diff_1_old = Pin_old - P1_old
P_diff_2_old = Pin_old - P2_old

# Flow rates (no scaling needed)
Q1_new = tracked_data_new[:, 4]
Q1_old = tracked_data_old[:, 3]
Q2_new = tracked_data_new[:, 3]
Q2_old = tracked_data_old[:, 4]

# Time or x-axis values
x = np.arange(len(Pin_new))

# Create a 2-row grid with 3 plots on top (pressures) and 2 on bottom (flow rates)
fig = plt.figure(figsize=(15, 10))

# Create GridSpec for more control over subplot layout
gs = fig.add_gridspec(2, 3)

# Top row: Pressure plots
ax1 = fig.add_subplot(gs[0, 0])  # Input Pressure
ax2 = fig.add_subplot(gs[0, 1])  # Output Pressure 1
ax3 = fig.add_subplot(gs[0, 2])  # Output Pressure 2

# Bottom row: Flow rate plots (centered)
ax4 = fig.add_subplot(gs[1, 0])  # Flow Rate 1
ax5 = fig.add_subplot(gs[1, 1])  # Flow Rate 2
ax6 = fig.add_subplot(gs[1, 2])  # Empty subplot


# Plot Input Pressure
ax1.plot(x, Pin_new, "b-", label="New Simulation")
ax1.plot(x, Pin_old, "r--", label="Old Simulation")
# ax1.plot(x, Pin_old_healthy, "g--", label="Old Simulation (Healthy)")
ax1.set_ylabel("Input Pressure")
ax1.set_title("Input Pressure Comparison")
ax1.grid(True)
ax1.legend()

# Plot Output Pressure 1
ax2.plot(x, P1_new, "b-", label="New Simulation")
ax2.plot(x, P1_old, "r--", label="Old Simulation")
ax2.set_ylabel("Output Pressure 1")
ax2.set_title("Output Pressure 1 Comparison")
ax2.grid(True)
ax2.legend()

# Plot Output Pressure 2
ax3.plot(x, P2_new, "b-", label="New Simulation")
ax3.plot(x, P2_old, "r--", label="Old Simulation")
ax3.set_ylabel("Output Pressure 2")
ax3.set_title("Output Pressure 2 Comparison")
ax3.grid(True)
ax3.legend()

# Plot Flow Rate 1
ax4.plot(x, Q1_new, "b-", label="New Simulation")
ax4.plot(x, Q1_old, "r--", label="Old Simulation")
ax4.set_ylabel("Flow Rate 1")
ax4.set_title("Flow Rate 1 Comparison")
ax4.grid(True)
ax4.legend()
ax4.set_xlabel("Time/Iteration")

# Plot Flow Rate 2
ax5.plot(x, P_diff_1_new, "b-", label="New Simulation")
ax5.plot(x, P_diff_1_old, "r--", label="Old Simulation")
ax5.set_ylabel("Pressure Difference 1")
ax5.set_title("Pressure Difference 1 Comparison")
ax5.grid(True)
ax5.legend()
ax5.set_xlabel("Time/Iteration")

ax6.plot(x, P_diff_2_new, "b-", label="New Simulation")
ax6.plot(x, P_diff_2_old, "r--", label="Old Simulation")
ax6.set_ylabel("Pressure Difference 2")
ax6.set_title("Pressure Difference 2 Comparison")
ax6.grid(True)
ax6.legend()
ax6.set_xlabel("Time/Iteration")

# ax6.plot(x, Pdiff_diseased, "r--", label="Diseased")
# ax6.plot(x, Pdiff_healthy, "g--", label="Healthy")
# ax6.set_ylabel("Pressure Difference")
# ax6.set_title("Pressure Difference Comparison")
# ax6.grid(True)
# ax6.legend()
# ax6.set_xlabel("Time/Iteration")
#
# Make subplots square-like
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.set_box_aspect(1)

plt.tight_layout()

# Save the figure with all subplots
plt.savefig(os.path.join(fig_dir, "pressure_flow_comparison.pdf"), bbox_inches="tight")

plt.show()
