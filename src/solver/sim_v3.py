import jax
import src.model.netlist_v7 as netlist
import jax.numpy as jnp


def create_time_step(
    size,
):
    def time_step_fn(carry, input):
        (
            prev_netlist,
            X_1,
            X_2,
            vessel_features,
            junction_features,
            inductor_indices,
            flowrate_query_indices,
            dt,
        ) = carry

        curr_Qin = input

        curr_netlist = netlist.update_element_values(
            prev_netlist, jnp.array([0]), jnp.array(curr_Qin)
        )

        G_curr, b_curr, updated_netlist = (
            netlist.assemble_matrices_with_non_linear_solve(
                curr_netlist,
                vessel_features,
                junction_features,
                size,
                inductor_indices,
                flowrate_query_indices,
                dt,
                X_1,
                X_2,
            )
        )
        X = jnp.linalg.solve(G_curr, b_curr)
        new_X1 = X
        new_X2 = X_1

        tracked_data = jnp.array(X[:, 0])
        # jax.debug.print("printing out tracked data {}", tracked_data)
        return (
            updated_netlist,
            new_X1,
            new_X2,
            vessel_features,
            junction_features,
            inductor_indices,
            flowrate_query_indices,
            dt,
        ), tracked_data

    return time_step_fn


def create_cardiac_simulation(init_carry, Qin, size, n_nodes, T, np1, dt, optim_idx):
    @jax.jit
    def cardiac_simulation():
        max_steps = int(np1 * T / dt)
        n_points_last_cycle = max_steps // np1 + 1
        time_step = create_time_step(size)
        _, tracked_data = jax.lax.scan(time_step, init_carry, Qin)
        # returns last cardiac cycle
        return tracked_data[-n_points_last_cycle:, :]

    return cardiac_simulation


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
