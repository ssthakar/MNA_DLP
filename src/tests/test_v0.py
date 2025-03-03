import os
import src.model.netlist_v4 as netlist
import time
import jax.numpy as jnp

current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_path = os.path.join(current_dir, "data", "elements.json")

test_netlist = netlist.create_netlist(data_file_path)

n_nodes = test_netlist.n_nodes
n_psources = test_netlist.n_psources
n_flowrates = test_netlist.n_flowrates
size = n_nodes + n_psources + n_flowrates
X_prev = jnp.zeros(size)
G_test, b_test = netlist.assemble_matrices(test_netlist, size, n_nodes, 0.001, X_prev)

for i in range(0, 10000):
    start_time = time.time()
    G_test, b_test = netlist.assemble_matrices(
        test_netlist, size, n_nodes, 0.001, X_prev
    )
    end_time = time.time()
    print(f"Iteration {i}: {end_time - start_time} seconds")
