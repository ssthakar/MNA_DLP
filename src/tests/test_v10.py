import os
import jax
import src.model.netlist_v9 as netlist
import src.solver.sim_v6 as sim
from src.utils.fft import fft_data
import time
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import optax
import shutil

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")

# FLAGS
RUN_INIT_SIM = True
PLOT_INIT_SIM = True
RUN_OPTIM = False
SHOW_PARAMS = True

start_time_total = time.time()
# paths to data
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, "data", "elements_v1.json")
all_files_path = os.path.join(current_dir, "data", "all_files")
output_path = "./output/"
junction_data_path = os.path.join(current_dir, "data", "junctions_v1.json")

# if os.path.exists(output_path):
#     shutil.rmtree(output_path)
# os.makedirs(output_path, exist_ok=True)
# print(f"Cleaned and recreated output directory: {output_path}")

# NOTE: this can be a function as well


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
np1 = int(15)
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
# Qin = Qin + 1e-3
Qin = Qin.flatten()

# Get initial matrices
G_test, b_test = netlist.assemble_matrices(
    test_netlist, size, inductor_indices, flowrate_query_indices, dt, X_1, X_2
)
np.savetxt(output_path + "/G_test_1.dat", G_test, fmt="%.4f")


target = jnp.array([125584.0, 125584.0])  # Example target value

optimizer = optax.adam(learning_rate=0.01)
params_in_phys_space = jnp.array([600, 0.0005, 29000, 600, 0.0005, 29000], float)

param_scale = jnp.array([100, 0.0001, 10000, 100, 0.0001, 10000], float)
params = params_in_phys_space / param_scale

optim_ids = jnp.array([7, 8, 9, 10, 11, 12], int)
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


if RUN_INIT_SIM:
    sim_start_time = time.time()
    simulation = sim.create_cardiac_simulation(
        init_carry, Qin, size, n_nodes, T, np1, dt, optim_ids
    )
    tracked_data = simulation()
    jax.block_until_ready(tracked_data)
    sim_end_time = time.time()
    sim_time = sim_end_time - sim_start_time
    np.savetxt(output_path + "tracked_data_C.dat", tracked_data, fmt="%.4f")
    np.savetxt(output_path + "target.dat", tracked_data, fmt="%.4f")
    if PLOT_INIT_SIM:
        plt.figure()
        plt.plot(tracked_data[:, 0] * 0.00075, "b-", label="P1 (mmHg)")
        plt.legend()
        plt.grid()
        plt.show()
    print(f"Initial Simulation completed in {sim_time:.3f} seconds.\n")


# create simulation function object with initial parameters
sim_with_initial_params = sim.create_cardiac_simulation_with_params(
    params_in_phys_space, init_carry, Qin, size, n_nodes, T, np1, dt, optim_ids
)
tracked_data_with_initial_params = sim_with_initial_params()
np.savetxt(
    output_path + "tracked_data_with_initial_params.dat",
    tracked_data_with_initial_params,
    fmt="%.8f",
)

target = np.loadtxt("./target.dat", dtype=float)
target = jnp.array(target, float)
print(target.shape)
plt.plot(target[:, 0])
plt.show()
plt.close()
# breakpoint()

max_optim_iter = 10000
print_per_iter = 10
simulate_per_iter = 10

loss_fn = sim.create_compute_loss(size, n_nodes, T, np1, dt, optim_ids)
if RUN_OPTIM:
    i = 0
    loss = netlist.LARGE_NUMBER
    start_time = time.time()
    print(f"Target Values: P_avg: {94:.3f}, P_max: {126}, P_min: {72}\n")
    print("********** Starting optimization **********")

    # Continue until either max iterations reached or loss threshold met
    while i < max_optim_iter and loss > 1e-3:
        loss, grads = jax.value_and_grad(loss_fn)(params, target, init_carry, Qin)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        jax.block_until_ready(loss)
        jax.block_until_ready(grads)
        end_time = time.time()

        if i % print_per_iter == 0:
            if i == 0:
                print(
                    f"itr# : {i}, Loss: {loss:.5f}, Time: {(end_time - start_time):.2f}, norm_grads: {jnp.linalg.norm(grads):.3f}, params norm: {jnp.linalg.norm(params * param_scale):.3f}"
                )
            else:
                print(
                    f"itr# : {i}, Loss: {loss:.5f}, Time: {(end_time - start_time) / print_per_iter:.2f}, norm_grads: {jnp.linalg.norm(grads):.3f}, params norm: {jnp.linalg.norm(params * param_scale):.3f}"
                )
            start_time = time.time()

        if i % simulate_per_iter == 0 and i > 0:
            print(f"Simulation at iter: {i}")
            sim_start_time = time.time()
            sim_with_current_params = sim.create_cardiac_simulation_with_params(
                params * param_scale,
                init_carry,
                Qin,
                size,
                n_nodes,
                T,
                np1,
                dt,
                optim_ids,
            )
            tracked_data_curr = sim_with_current_params()
            jax.block_until_ready(tracked_data_curr)
            sim_end_time = time.time()
            sim_time = sim_end_time - sim_start_time
            params_in_phys_space = params * param_scale
            np.savetxt(
                output_path + f"params_at_iter_{i}.dat",
                params_in_phys_space,
                fmt="%.9f",
            )
            if SHOW_PARAMS:
                print(f"params at iter {i}: {params_in_phys_space}")
            p1_avg = jnp.mean(tracked_data_curr[:, 0]) * 0.00075
            p1_max = jnp.max(tracked_data_curr[:, 0]) * 0.00075
            p1_min = jnp.min(tracked_data_curr[:, 0]) * 0.00075
            q1_avg = jnp.mean(tracked_data_curr[:, 3])
            q2_avg = jnp.mean(tracked_data_curr[:, 4])
            plt.figure()
            plt.plot(tracked_data_curr[:, 0] * 0.00075, "b-", label="Pin (curr) (mmHg)")
            plt.plot(
                target[:, 0] * 0.00075,
                "r-",
                label="Pin (target) (mmHg)",
            )
            plt.legend()
            plt.grid()
            plt.savefig(output_path + f"p1_iter_{i}.png")
            target_to_plot = jnp.array([p1_avg, p1_max, p1_min, q1_avg, q2_avg])
            np.savetxt(
                output_path + f"tracked_data_iter_{i}_sim.dat",
                tracked_data_curr,
                fmt="%.4f",
            )
            np.savetxt(output_path + f"target_iter_{i}.dat", target_to_plot, fmt="%.4f")
            print(f"P_avg: {p1_avg:.3f}, P_max: {p1_max:.3f}, P_min: {p1_min:.3f}\n")

        i += 1

    # Run final simulation and save optimized parameters
    print(f"\n********** Optimization finished **********")
    print(f"Stopped after {i} iterations with final loss: {loss:.5f}")

    # Convert parameters to physical space
    params_in_phys_space = params * param_scale

    # Save optimized parameters
    np.savetxt(
        output_path + "final_optimized_params.dat",
        params_in_phys_space,
        fmt="%.9f",
    )

    # Print final optimized parameters
    print("\nFinal optimized parameters:")
    print(params_in_phys_space)

    # Run final simulation with optimized parameters
    print("\nRunning final simulation with optimized parameters...")
    final_sim = sim.create_cardiac_simulation_with_params(
        params_in_phys_space,
        init_carry,
        Qin,
        size,
        n_nodes,
        T,
        np1,
        dt,
        optim_ids,
    )
    final_tracked_data = final_sim()

    # Calculate and print final metrics
    p1_avg = jnp.mean(final_tracked_data[:, 0]) * 0.00075
    p1_max = jnp.max(final_tracked_data[:, 0]) * 0.00075
    p1_min = jnp.min(final_tracked_data[:, 0]) * 0.00075
    q1_avg = jnp.mean(final_tracked_data[:, 3])
    q2_avg = jnp.mean(final_tracked_data[:, 4])

    print(
        f"Final results - P_avg: {p1_avg:.3f}, P_max: {p1_max:.3f}, P_min: {p1_min:.3f}"
    )

    # Save final tracked data
    np.savetxt(output_path + "final_tracked_data.dat", final_tracked_data, fmt="%.4f")

    # Plot final results
    plt.figure(figsize=(10, 6))
    plt.plot(final_tracked_data[:, 0] * 0.00075, "b-", label="Final P1 (mmHg)")
    plt.plot(
        tracked_data_with_initial_params[:, 0] * 0.00075,
        "r-",
        label="Initial P1 (mmHg)",
    )
    plt.legend()
    plt.grid()
    plt.title("Initial vs Final Pressure")
    plt.xlabel("Time Steps")
    plt.ylabel("Pressure (mmHg)")
    plt.savefig(output_path + "final_pressure_comparison.png")
