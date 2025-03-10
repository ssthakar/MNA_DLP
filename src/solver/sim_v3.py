import jax
from numpy import int32
import src.model.netlist_v7 as netlist
import jax.numpy as jnp
from functools import partial


# NOTE: Extremely shitty implementation, need to make this more general and faster, JAX Shit is all over the place, function closures etc
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
        # jax.debug.print(
        #     "printing out optim values {}",
        #     curr_netlist.element_values[optim_element_idx],
        # )
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
            curr_netlist.element_values[9],
            optim_nodes[0, 0],
            optim_nodes[0, 1],
        )
        q2 = netlist.get_flowrate_at_resistor(
            X,
            curr_netlist.element_values[10],
            optim_nodes[3, 0],
            optim_nodes[3, 1],
        )
        # p1 = X[optim_nodes[0, 0] - 1, 0]

        # we are tracking pressure at the inlet of the idealized bifurcation
        p1 = X[0, 0]
        # p2 = X[optim_nodes[3, 0] - 1, 0]

        # jax.debug.print(
        #     "printing out element values {}",
        #     curr_netlist.element_values[optim_element_idx],
        # )
        #
        # jax.debug.print(
        #     "printing out nodes {} {}", optim_nodes[0, 0], optim_nodes[3, 0]
        # )

        # tracked_data = jnp.array([p1, p2])
        # jax.debug.print("{} {}", q1 + q2, curr_Qin)

        # jax.debug.print("qq shape {} {}", p1, optim_nodes[0, 0])
        tracked_data = jnp.array([p1, q1[0], q2[0]])
        # for plotting, not required in actual solver
        return (
            updated_netlist,
            new_X1,
            new_X2,
            vessel_features,
            dt,
        ), tracked_data

    return time_step_fn


def create_cardiac_simulation(init_carry, Qin, size, n_nodes, T, np1, dt, optim_idx):
    @jax.jit
    def cardiac_simulation():
        max_steps = int(np1 * T / 0.01)
        n_points_last_cycle = max_steps // np1 + 1
        time_step = create_time_step(size, n_nodes, optim_idx)
        _, tracked_data = jax.lax.scan(time_step, init_carry, Qin)
        # return tracked_data
        return tracked_data[-n_points_last_cycle:, :]  # returns last cardiac cycle

    return cardiac_simulation


# tranforming parameters from optim to phys space and vice versa
def transform_to_optim_space(
    params,
    param_scales,
):
    params = jnp.array(params, dtype=jnp.float32)
    params_to_optim_space = params / param_scales
    return params_to_optim_space


def transform_to_phys_space(params, param_scales):
    params = jnp.array(params, dtype=jnp.float32)
    params_to_phys_space = params * param_scales
    return params_to_phys_space


def create_compute_loss(size, n_nodes, T, np_1, dt, optim_idx):
    @jax.jit
    def compute_loss(params, target, init_carry, Qin):
        params = jnp.array(params, dtype=jnp.float32)
        # hard coded for now, but will create functions to make this more general
        param_scales = jnp.array([100, 10000, 0.0001, 100, 10000, 0.0001], float)
        params_in_phys_space = transform_to_phys_space(params, param_scales)
        curr_netlist, X_1, X_2, vessel_features, dt = init_carry
        netlist_with_params = netlist.update_element_values(
            curr_netlist, optim_idx, params_in_phys_space
        )
        #
        # jax.debug.print(
        #     "printing out optim values {}", netlist_with_params.element_values[11]
        # )
        updated_carry = (netlist_with_params, X_1, X_2, vessel_features, dt)

        cardiac_simulation = create_cardiac_simulation(
            updated_carry,
            Qin,
            size,
            n_nodes,
            T,
            np_1,
            dt,
            optim_idx,
        )

        tracked_data = cardiac_simulation()
        p1_avg = jnp.mean(tracked_data[:, 0]) * 0.00075
        p1_max = jnp.max(tracked_data[:, 0]) * 0.00075
        p1_min = jnp.min(tracked_data[:, 0]) * 0.00075
        # jax.debug.print("p1_avg {}", p1_avg)
        q1_avg = jnp.mean(tracked_data[:, 1])

        q2_avg = jnp.mean(tracked_data[:, 2])
        target_systolic = 126.0211
        target_diastolic = 72.04
        target_mean = 94.27
        lagrangian_multiplier = 0.2
        p1_loss = jnp.mean((p1_avg - target_mean) ** 2)
        p1_sys = jnp.mean((p1_max - target_systolic) ** 2)
        p1_dia = jnp.mean((p1_min - target_diastolic) ** 2)
        q_loss = jnp.mean((q1_avg - q2_avg) ** 2) * lagrangian_multiplier
        # jax.debug.print("p1_loss {} q_loss {}", p1_loss, q_loss)
        loss = p1_loss + q_loss + p1_sys + p1_dia
        # loss = p1_loss + p2_loss
        return loss

    return compute_loss


def create_optimizer_step(optimizer):
    @jax.jit
    def optimizer_step(optimizer, data):
        pass

    return optimizer_step
