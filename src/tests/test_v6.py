import os
import jax
import src.model.netlist_v7 as netlist
import src.solver.sim_v2 as sim
from src.utils.fft import fft_data
import time
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import optax


jax.config.update("jax_enable_x64", True)

# paths to data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, "data", "elements_v0.json")
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
dt = 0.01

# Boundary conditions
aorta_flow = np.loadtxt(aorta_flow_data_path, dtype=float)
Q_inll = fft_data(T, dt, aorta_flow)
Qin = np.zeros((int(np1 * T / dt) + 1, 1))
Qin[0 : int(T / dt + 1), 0] = Q_inll[:, 0]
for i in range(0, np1):
    Qin[int(i * T / dt + 1) : int(i * T / dt + 1 + T / dt), 0] = Q_inll[1:, 0]
Qin = jnp.array(Qin, float)
Qin = Qin.flatten()
print(Qin.shape)
# Pin = np.zeros_like(Qin)
plt.plot(Qin[-1100:])
plt.grid(True)
plt.show()

# Get initial matrices
G_test, b_test = netlist.assemble_matrices(test_netlist, size, n_nodes, dt, X_1, X_2)
np.savetxt(output_path + "/G_test_1.dat", G_test, fmt="%.4f")

# Define fixed point iteration parameters
max_iter = 10
tol = 1e-5
# Define optimizer parameters
target = jnp.array([125489.0, 125489.0])  # Example target value
optimizer = optax.adam(1e-5)
params = jnp.array([0.0001, 0.0001])  # Example parameter values (to be optimized)

optim_ids = jnp.array([11, 12], int)
opt_state = optimizer.init(params)
time_step = sim.create_time_step(size, n_nodes, optim_ids)

init_carry = (test_netlist, X_1, X_2, vessel_features, dt)

simulation = sim.create_cardiac_simulation(
    init_carry, Qin, size, n_nodes, T, np1, dt, optim_ids
)
total_start_time = time.time()

Pin = simulation()
np.savetxt(output_path + "/Pin_1.dat", Pin, fmt="%.4f")
cycle_data_p1_list = []
loss_fn = sim.create_compute_loss(size, n_nodes, T, np1, dt, optim_ids)
for i in range(1000):
    # print(f"printing param {params}")
    start_time = time.time()
    # loss = sim.compute_loss(
    #     params, target, init_carry, Qin, size, n_nodes, T, np1, dt, optim_ids
    # )

    loss, grads = jax.value_and_grad(loss_fn)(params, target, init_carry, Qin)

    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    jax.block_until_ready(loss)
    jax.block_until_ready(grads)
    end_time = time.time()
    print(
        f"Loss: {loss}, grads:{grads}, Time: {end_time - start_time},params: {params}"
    )


# breakpoint()

# final_carry, Pin = jax.lax.scan(time_step, init_carry, Qin)
jax.block_until_ready(Pin)
total_end_time = time.time()

print(f"Total simulation time: {total_end_time - total_start_time} seconds")
print(jnp.mean(Pin[:, 0:1]))

# @partial(jax.jit, static_argnums=(2, 3))
# def run_simulation(init_carry, Qin, size, n_nodes):
#     time_step_fn = sim.create_time_step(size, n_nodes)
#     return jax.lax.scan(time_step_fn, init_carry, Qin)
#
#
# start_time = time.time()
# final_carry, Pin = run_simulation(init_carry, Qin, size, n_nodes)
# jax.block_until_ready(Pin)
# end_time = time.time()
# print(f"Total simulation time: {end_time - start_time} seconds")

plt.figure(figsize=(10, 6))
# plt.plot(Pin[:, 1:2] + Pin[:, 2:3], label="Pressure at inlet")
plt.plot(Pin[:, 0:1], label="Pressure at inlet")
# plt.plot(Qin[-111:], label="Flow rate at inlet")
plt.title("Pressure over Time")
plt.xlabel("Time Steps")
plt.ylabel("Pressure")
plt.grid(True)

plt.show()
