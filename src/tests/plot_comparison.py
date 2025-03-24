import matplotlib.pyplot as plt
import numpy as np
import os

fig_dir = "figures"
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)
tracked_data_new = np.loadtxt("./tracked_data_new.dat")
tracked_data_old = np.loadtxt("./tracked_data_old.dat")
tracked_data_old = tracked_data_old[-1101:, :]

# tracked_data_old_healthy = np.loadtxt("./tracked_data_old_healthy.dat")
# tracked_data_old_healthy = tracked_data_old_healthy[-1101:, :]
#
Pin_new = tracked_data_new[:, 0] * 0.00075
Pin_old = tracked_data_old[:, 0] * 0.00075
Pin_new_avg = np.mean(Pin_new)
print(f"New simulation average input pressure: {Pin_new_avg}")
Pin_old_avg = np.mean(Pin_old)
print(f"Old simulation average input pressure: {Pin_old_avg}")
# Pin_old_healthy = tracked_data_old_healthy[:, 0] * 0.00075


P1_new = tracked_data_new[:, 1] * 0.00075
P1_new_avg = np.mean(P1_new)
print(f"New simulation average pressure 1: {P1_new_avg}")
P1_old = tracked_data_old[:, 1] * 0.00075
P1_old_avg = np.mean(P1_old)
print(f"Old simulation average pressure 1: {P1_old_avg}")
# P1_old_healthy = tracked_data_old_healthy[:, 1] * 0.00075

P2_new = tracked_data_new[:, 2] * 0.00075
P2_new_avg = np.mean(P2_new)
print(f"New simulation average pressure 2: {P2_new_avg}")
P2_old = tracked_data_old[:, 2] * 0.00075
P2_old_avg = np.mean(P2_old)
print(f"Old simulation average pressure 2: {P2_old_avg}")
# P2_old_healthy = tracked_data_old_healthy[:, 2] * 0.00075

Pdiff_diseased = Pin_old - P2_old

# Pdiff_healthy = Pin_old_healthy - P2_old_healthy
# Flow rates (no scaling needed)
Q1_new = tracked_data_new[:, 4]
Q1_new_avg = np.mean(Q1_new)
print(f"New simulation average flow rate 1: {Q1_new_avg}")
Q1_old = tracked_data_old[:, 3]
Q1_old_avg = np.mean(Q1_old)
print(f"Old simulation average flow rate 1: {Q1_old_avg}")
Q2_new = tracked_data_new[:, 3]
Q2_new_avg = np.mean(Q2_new)
print(f"New simulation average flow rate 2: {Q2_new_avg}")
Q2_old = tracked_data_old[:, 4]
Q2_old_avg = np.mean(Q2_old)
print(f"Old simulation average flow rate 2: {Q2_old_avg}")
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
ax5.plot(x, Q2_new, "b-", label="New Simulation")
ax5.plot(x, Q2_old, "r--", label="Old Simulation")
ax5.set_ylabel("Flow Rate 2")
ax5.set_title("Flow Rate 2 Comparison")
ax5.grid(True)
ax5.legend()
ax5.set_xlabel("Time/Iteration")

# ax6.plot(x, Pdiff_diseased, "r--", label="Diseased")
# ax6.plot(x, Pdiff_healthy, "g--", label="Healthy")
ax6.set_ylabel("Pressure Difference")
ax6.set_title("Pressure Difference Comparison")
ax6.grid(True)
ax6.legend()
ax6.set_xlabel("Time/Iteration")
# Make subplots square-like
for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.set_box_aspect(1)
plt.tight_layout()
# Save the figure with all subplots
plt.savefig(os.path.join(fig_dir, "pressure_flow_comparison.pdf"), bbox_inches="tight")
plt.show()
#
# q_total = Q1_new + Q2_new
# qin = np.loadtxt("./Qin.dat")
# plt.plot(q_total, "b-", label="Total Flow Rate")
# plt.plot(qin[-1101:,], "r--", label="Input Flow Rate")
# plt.legend()
# plt.show()
