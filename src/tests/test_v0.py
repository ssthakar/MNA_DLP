import os
import jax
import src.model.netlist_v4 as netlist
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

test_netlist = netlist.create_netlist(data_file_path)

# NOTE: until helper function, set manually
vessel_ids = jnp.array([1, 3, 6], int)
cumsum_array = jnp.array([[0], [20], [40], [60]], int)
acl_data_path = os.path.join(all_files_path, "area_curv_length.dat")
aorta_flow_data_path = os.path.join(all_files_path, "aorta-flow.dat")
vessel_features = netlist.read_acl_features(acl_data_path, vessel_ids, cumsum_array)
resistor_nodes = test_netlist.nodes[vessel_ids]  # will need this for flow rate


# sizing
n_nodes = test_netlist.n_nodes
n_psources = test_netlist.n_psources
n_flowrates = test_netlist.n_flowrates
size = n_nodes + n_psources + n_flowrates


# init
initial_pressure = 100 / 0.00075
X_prev = jnp.ones((size, 1))
X_prev = X_prev * initial_pressure
X_prev = X_prev.at[-3:, 0].set(0.1)

# time controls
T = 1.1
np1 = int(10)
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
# plt.plot(Qin)
# plt.show()


G_test, b_test = netlist.assemble_matrices(test_netlist, size, n_nodes, dt, X_prev)
np.savetxt(output_path + "G_test_1.dat", G_test, fmt="%.4f")
# print(b_test)
# breakpoint()

prev_netlist = test_netlist
for c in range(0, int(np1 * T / dt) + 1):
    start_time = time.time()
    curr_netlist = netlist.update_element_values(
        prev_netlist, jnp.array([0]), jnp.array(Qin[c, 0])
    )
    G_curr, b_curr = netlist.assemble_matrices(curr_netlist, size, n_nodes, dt, X_prev)
    # print(jnp.linalg.cond(G_curr))
    X = jnp.linalg.solve(G_curr, b_curr)
    end_time = time.time()
    print(f"Iteration {c}: {end_time - start_time} seconds")
    # print(b_curr, G_curr)
    if jnp.isnan(b_curr).any():
        print("nan")
        break
    X_prev = X
    prev_netlist = curr_netlist
    Pin[c, 0] = X[0, 0]

plt.plot(Pin)
plt.show()
np.savetxt(output_path + "Pin_transient.dat", Pin)
np.savetxt(output_path + "G_test_1.dat", G_test, fmt="%.4f")
np.savetxt(output_path + "b_test_1.dat", b_test, fmt="%.4f")

# for i in range(0, 10000):
#     start_time = time.time()
#     G_test, b_test = netlist.assemble_matrices(
#         test_netlist, size, n_nodes, 0.01, X_prev
#     )
#     end_time = time.time()
#     print(f"Iteration {i}: {end_time - start_time} seconds")
