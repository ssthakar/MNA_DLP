import jax
import src.model.netlist_v7 as netlist
import jax.numpy as jnp
from functools import partial


# implement closure to keep system size and n_nodes static through traces
def create_time_step(size, n_nodes):
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
        pressure_inlet = X[0, 0]
        return (
            updated_netlist,
            new_X1,
            new_X2,
            vessel_features,
            dt,
        ), pressure_inlet

    return time_step_fn


@partial(jax.jit, static_argnums=(2, 3))
def cardiac_simulation(init_carry, Qin, size, n_nodes, T, np1, dt):
    max_steps = int(np1 * T / dt) + 1
    points_per_period = max_steps // np1
    cycle_data = jnp.zeros((points_per_period, 4))

    time_step = create_time_step(size, n_nodes)

    _, pressure_inlet = jax.lax.scan(time_step, init_carry, Qin)

    return pressure_inlet


def compute_loss(params, target):
    pass


def create_optimizer_step(optimizer):
    @jax.jit
    def optimizer_step(optimizer, data):
        pass

    return optimizer_step
