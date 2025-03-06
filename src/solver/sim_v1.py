import jax
from numpy import int32
import src.model.netlist_v7 as netlist
import jax.numpy as jnp
from functools import partial


# implement closure to keep system size and n_nodes static through traces
def create_time_step(
    size,
    n_nodes,
    optim_element_idx,
):
    # NOTE: applying jax.jit here is redundant as jax.lax.scan is primitive
    def time_step_fn(carry, x):
        prev_netlist, X_1, X_2, vessel_features, dt = carry
        curr_Qin = x

        # TODO: implement a function to make applying boundary conditions more general through current/pressure sources
        curr_netlist = netlist.update_element_values(
            prev_netlist, jnp.array([0]), jnp.array(curr_Qin)
        )
        G_curr, b_curr, updated_netlist = (
            netlist.assemble_matrices_with_non_linear_solve(
                curr_netlist, vessel_features, size, n_nodes, dt, X_1, X_2
            )
        )
        X = jnp.linalg.solve(G_curr, b_curr)
        new_X1 = X
        new_X2 = X_1
        optim_nodes = curr_netlist.nodes[optim_element_idx]
        q1 = netlist.get_flowrate_at_resistor(
            X,
            curr_netlist.element_values[optim_element_idx[0]],
            optim_nodes[0, 0],
            optim_nodes[0, 1],
        )
        q2 = netlist.get_flowrate_at_resistor(
            X,
            curr_netlist.element_values[optim_element_idx[1]],
            optim_nodes[1, 0],
            optim_nodes[1, 1],
        )
        p1 = X[optim_nodes[0, 0] - 1, 0]
        tracked_data = jnp.array([p1, q1[0], q2[0]])
        # jax.debug.print("{} {}", q1 + q2, curr_Qin)

        # jax.debug.print("qq shape {} {}", p1, optim_nodes[0, 0])
        # tracked_data = jnp.array([p1, q1, q2])
        # for plotting, not required in actual solver
        return (
            updated_netlist,
            new_X1,
            new_X2,
            vessel_features,
            dt,
        ), tracked_data

    return time_step_fn


def create_time_step_for_optim(size, n_nodes, optim_element_idx):
    # NOTE: applying jax.jit here is redundant as jax.lax.scan is primitive
    def time_step_fn(carry, x):
        prev_netlist, X_1, X_2, vessel_features, dt = carry
        curr_Qin = x

        # TODO: implement a function to make applying boundary conditions more general through current/pressure sources
        curr_netlist = netlist.update_element_values(
            prev_netlist, jnp.array([0]), jnp.array(curr_Qin)
        )
        G_curr, b_curr, updated_netlist = (
            netlist.assemble_matrices_with_non_linear_solve(
                curr_netlist, vessel_features, size, n_nodes, dt, X_1, X_2
            )
        )
        X = jnp.linalg.solve(G_curr, b_curr)
        new_X1 = X
        new_X2 = X_1
        # for plotting, not required in actual solver
        pressure_inlet = X[5, 0]
        return (
            updated_netlist,
            new_X1,
            new_X2,
            vessel_features,
            dt,
        ), pressure_inlet

    return time_step_fn


@partial(jax.jit, static_argnums=(2, 3))
def cardiac_simulation(init_carry, Qin, size, n_nodes, T, np1, dt, optim_idx):
    max_steps = int(np1 * T / dt) + 1
    points_per_period = max_steps // np1
    time_step = create_time_step(size, n_nodes, optim_idx)
    _, tracked_data = jax.lax.scan(time_step, init_carry, Qin)
    tracked_data = tracked_data[-1100:,]
    return tracked_data


def compute_loss(params, target):
    pass


def create_optimizer_step(optimizer):
    @jax.jit
    def optimizer_step(optimizer, data):
        pass

    return optimizer_step
