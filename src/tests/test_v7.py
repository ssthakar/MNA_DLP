import os
import jax
import src.model.netlist_v7 as netlist
import src.solver.sim_v3 as sim
from src.utils.fft import fft_data
import time
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax


jax.config.update("jax_enable_x64", True)

# paths to data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, "data", "elements_v1.json")
all_files_path = os.path.join(current_dir, "data", "all_files")
output_path = "./output"
junction_data_path = os.path.join(current_dir, "data", "junctions_v1.json")

# NOTE: this can be a function as well
# create netlist and supporting data structures
test_netlist = netlist.create_netlist(data_file_path)
# print(test_netlist)
# breakpoint()
inductor_indices = netlist.create_inductor_index_mask(test_netlist)
flowrate_query_indices = netlist.create_flowrate_query_indices(inductor_indices)
# Create netlist

# NOTE: until helper function, set manually
vessel_ids = jnp.array([1, 3, 5], int)
cumsum_array = jnp.array([[0], [20], [40], [60]], int)
acl_data_path = os.path.join(all_files_path, "area_curv_length.dat")
aorta_flow_data_path = os.path.join(all_files_path, "aorta-flow.dat")
vessel_features = netlist.create_vessel_features(
    acl_data_path, vessel_ids, cumsum_array, test_netlist
)


# NOTE: club this with the initial construction of the netlist function
junction_features = netlist.create_junction_features(junction_data_path)
resistor_nodes = test_netlist.nodes[vessel_ids]  # will need this for flow rate

# sizing
n_nodes = test_netlist.n_nodes
n_psources = test_netlist.n_psources
n_flowrates = test_netlist.n_flowrates
size = n_nodes + n_psources + n_flowrates

# NOTE: this can be a function for any general simulation, specify P_init and Q_init
initial_pressure = 100 / 0.00075
X_1 = jnp.ones((size, 1))
X_1 = X_1 * initial_pressure
X_1 = X_1.at[-3:, 0].set(0.1)
X_2 = X_1

# time controls
T = 1.1
np1 = int(10)
dt = 0.01


# NOTE: this can be a function for any general waveform
# Boundary conditions
aorta_flow = np.loadtxt(aorta_flow_data_path, dtype=float)
Q_inll = fft_data(T, dt, aorta_flow)
Qin = np.zeros((int(np1 * T / dt) + 1, 1))
Qin[0 : int(T / dt + 1), 0] = Q_inll[:, 0]
for i in range(0, np1):
    Qin[int(i * T / dt + 1) : int(i * T / dt + 1 + T / dt), 0] = Q_inll[1:, 0]
Qin = jnp.array(Qin, float)
Qin = Qin.flatten()

# Get initial matrices
G_test, b_test = netlist.assemble_matrices(
    test_netlist, size, inductor_indices, flowrate_query_indices, dt, X_1, X_2
)
np.savetxt(output_path + "/G_test_1.dat", G_test, fmt="%.4f")


target = jnp.array([125584.0, 125584.0])  # Example target value

optimizer = optax.adam(learning_rate=0.1)
params_in_phys_space = jnp.array([600, 29000, 0.0005, 600, 29000, 0.0005], float)

param_scale = jnp.array([100, 10000, 0.0001, 100, 10000, 0.0001], float)
params = params_in_phys_space / param_scale

optim_ids = jnp.array([5, 9, 11, 8, 10, 12], int)
opt_state = optimizer.init(params)
time_step = sim.create_time_step(size)

init_carry = (
    test_netlist,
    X_1,
    X_2,
    vessel_features,
    junction_features,
    inductor_indices,
    flowrate_query_indices,
    dt,
)

simulation = sim.create_cardiac_simulation(
    init_carry, Qin, size, n_nodes, T, np1, dt, optim_ids
)
total_start_time = time.time()

tracked_data = simulation()
print("done")
np.savetxt(output_path + "/last_cycle_unoptimized.dat", tracked_data, fmt="%.4f")
cycle_data_p1_list = []
loss_fn = sim.create_compute_loss(size, n_nodes, T, np1, dt, optim_ids)
for i in range(0):
    start_time = time.time()
    loss, grads = jax.value_and_grad(loss_fn)(params, target, init_carry, Qin)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    jax.block_until_ready(loss)
    jax.block_until_ready(grads)
    end_time = time.time()
    # print(
    #     f"Loss: {loss}, grads:{grads}, Time: {end_time - start_time},params: {params * param_scale}"
    # )
    if i % 10 == 0:
        print(
            f"Iteration : {i}, Loss: {loss:.3f}, grads: {[f'{g:.3f}' for g in grads]}, Time: {end_time - start_time:.3f}, params: {[f'{p:.6f}' for p in (params * param_scale)]}"
        )


# breakpoint()

# final_carry, Pin = jax.lax.scan(time_step, init_carry, Qin)
jax.block_until_ready(tracked_data)
total_end_time = time.time()

print(f"Total simulation time: {total_end_time - total_start_time} seconds")
print("tracked data shape", tracked_data.shape)
print(jnp.mean(tracked_data[:, 0:1]))
print(jnp.mean(tracked_data[:, 1:2]))

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
Pin = tracked_data[:, 0:1]
Pin_scaled = Pin * 0.00075
print(f"systolic pressure: {Pin_scaled.max()}")
print(f"diastolic pressure: {Pin_scaled.min()}")
print(f"mean pressure: {Pin_scaled.mean()}")

plt.figure(figsize=(10, 6))
# plt.plot(Pin[:, 1:2] + Pin[:, 2:3], label="Pressure at inlet")
plt.plot(tracked_data[:, 0:1] * 0.00075, label="Pressure at inlet")
# plt.plot(Qin[-111:], label="Flow rate at inlet")
plt.title("Pressure over Time")
plt.xlabel("Time Steps")
plt.ylabel("Pressure")
plt.grid(True)

plt.show()
