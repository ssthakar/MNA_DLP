import os
import jax
import src.model.netlist_v7 as netlist
from src.utils.fft import fft_data
import time
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# paths to data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, "data", "elements.json")
all_files_path = os.path.join(current_dir, "data", "all_files")
output_path = "./output"

# Create netlist
test_netlist = netlist.create_netlist(data_file_path)

# NOTE: until helper function, set manually
vessel_ids = jnp.array([1, 3, 6], int)
cumsum_array = jnp.array([[0], [20], [40], [60]], int)
acl_data_path = os.path.join(all_files_path, "area_curv_length.dat")
aorta_flow_data_path = os.path.join(all_files_path, "aorta-flow.dat")
vessel_features = netlist.create_vessel_features(
    acl_data_path, vessel_ids, cumsum_array, test_netlist
)
resistor_nodes = test_netlist.nodes[vessel_ids]  # will need this for flow rate

# sizing
n_nodes = test_netlist.n_nodes
n_psources = test_netlist.n_psources
n_flowrates = test_netlist.n_flowrates
size = n_nodes + n_psources + n_flowrates

# init
initial_pressure = 100 / 0.00075
X_1 = jnp.ones((size, 1))
X_1 = X_1 * initial_pressure
X_1 = X_1.at[-3:, 0].set(0.1)
X_2 = X_1

# time controls
T = 1.1
np1 = int(20)
dt = 0.001

# Boundary conditions
aorta_flow = np.loadtxt(aorta_flow_data_path, dtype=float)
Q_inll = fft_data(T, dt, aorta_flow)
Qin = np.zeros((int(np1 * T / dt) + 1, 1))
Qin[0 : int(T / dt + 1), 0] = Q_inll[:, 0]
for i in range(0, np1):
    Qin[int(i * T / dt + 1) : int(i * T / dt + 1 + T / dt), 0] = Q_inll[1:, 0]
Qin = jnp.array(Qin, float)
print(Qin.shape)
Pin = np.zeros_like(Qin)

# Get initial matrices
G_test, b_test = netlist.assemble_matrices(test_netlist, size, n_nodes, dt, X_1, X_2)
np.savetxt(output_path + "/G_test_1.dat", G_test, fmt="%.4f")

# Define fixed point iteration parameters
max_iter = 10
tol = 1e-5

# Main simulation loop with integrated non-linear solution
prev_netlist = test_netlist
for c in range(0, int(np1 * T / dt) + 1):
    start_time = time.time()

    # Update flow source value
    curr_netlist = netlist.update_element_values(
        prev_netlist, jnp.array([0]), jnp.array(Qin[c, 0])
    )

    # Use the assemble_matrices_non_linear function which integrates fixed point iteration
    G_curr, b_curr, updated_netlist = netlist.assemble_matrices_with_non_linear_solve(
        curr_netlist, vessel_features, size, n_nodes, dt, X_1, X_2, max_iter, tol
    )

    # Solve the system with the assembled matrices
    X = jnp.linalg.solve(G_curr, b_curr)

    end_time = time.time()
    print(f"Iteration {c}: {end_time - start_time} seconds")

    if jnp.isnan(b_curr).any() or jnp.isnan(X).any():
        print("nan detected")
        break
    X_2 = X_1
    X_1 = X

    prev_netlist = updated_netlist
    Pin[c, 0] = X[0, 0]

# Plot and save results
plt.figure(figsize=(10, 6))
plt.plot(Pin)
plt.title("Pressure at Node 1 over Time")
plt.xlabel("Time Steps")
plt.ylabel("Pressure")
plt.grid(True)
plt.show()

# Save results
np.savetxt(output_path + "/Pin_transient_non_linear_1e-3.dat", Pin)
np.savetxt(output_path + "/G_test_1.dat", G_test, fmt="%.4f")
np.savetxt(output_path + "/b_test_1.dat", b_test, fmt="%.4f")
