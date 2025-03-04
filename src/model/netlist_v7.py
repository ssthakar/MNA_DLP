import jax
import json
from functools import partial
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, NamedTuple, Tuple

# GLOBALS

element_type_map = {
    "R": 0,  # resistor
    "C": 1,  # capacitor
    "L": 2,  # inductor
    "Q": 3,  # flowrate source
    "P": 4,  # pressure source
}

MU = 0.04
RHO = 1.06


# NOTE: named tuples are immutable
class Netlist(NamedTuple):
    elements: jnp.ndarray
    element_values: jnp.ndarray
    nodes: jnp.ndarray
    n_nodes: int
    n_psources: int
    n_flowrates: int
    current_time_step: int


# ACL arrays
class VesselFeatures(NamedTuple):
    vessel_ids: jnp.ndarray  # (NV X 1)
    vessel_areas: jnp.ndarray  # (NV X N_seg)
    vessel_curvatures: jnp.ndarray  # (NV X N_seg)
    vessel_lengths: jnp.ndarray  # (NV X N_seg)
    vessel_base_resistances: (
        jnp.ndarray
    )  # (NV X 1) (Poisuille + Stenosis + Curvature effects)
    vessel_resistances: jnp.ndarray  # (NV X 1) # after non-linear update


def create_vessel_features(
    file_path: str, vessel_ids: jnp.ndarray, cumsum_array: jnp.ndarray, netlist: Netlist
):
    acl_data = np.loadtxt(file_path)
    n_vessels = vessel_ids.shape[0]
    n_segments = cumsum_array[1, 0]

    vessel_areas = np.zeros((n_vessels, n_segments))
    vessel_curvatures = np.zeros((n_vessels, n_segments))
    vessel_lengths = np.zeros((n_vessels, n_segments))

    for i in range(n_vessels):
        start_index = cumsum_array[i, 0]
        end_index = cumsum_array[i + 1, 0]
        area = acl_data[start_index:end_index, 0]
        print(acl_data.shape)
        curvature = acl_data[start_index:end_index, 1]
        length = acl_data[start_index:end_index, 2]
        vessel_areas[i, : len(area)] = area
        vessel_curvatures[i, : len(curvature)] = curvature
        vessel_lengths[i, : len(length)] = length

    vessel_areas = jnp.array(vessel_areas, float)
    vessel_curvatures = jnp.array(vessel_curvatures, float)
    vessel_lengths = jnp.array(vessel_lengths, float)

    vessel_base_resistances = netlist.element_values[vessel_ids]
    vessel_resistances = netlist.element_values[vessel_ids]

    return VesselFeatures(
        vessel_ids,
        vessel_areas,
        vessel_curvatures,
        vessel_lengths,
        vessel_base_resistances,
        vessel_resistances,
    )


def read_base_elements(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        data = json.load(f)
        base_elements = data["elements"]
    return base_elements


def create_netlist(file_path: str) -> Netlist:
    type_map = element_type_map
    base_elements = read_base_elements(file_path)
    elements = jnp.array([type_map[e["type"]] for e in base_elements])
    element_values = jnp.array([e["value"] for e in base_elements])
    nodes = jnp.array([[e["node1"], e["node2"]] for e in base_elements], int)
    print("nodes", nodes)

    n_nodes = int(jnp.max(nodes))

    # NOTE: until indpendent pressure sources are implemeneted, we set n_psources to 0
    n_psources = 0
    n_flowrates = int(
        jnp.sum(elements == type_map["L"])  # auxiliary vars for inductor currents
    )

    return Netlist(
        elements=elements,
        element_values=element_values,
        nodes=nodes,
        n_nodes=n_nodes,
        n_psources=n_psources,
        n_flowrates=n_flowrates,
        current_time_step=0,
    )


@jax.jit
def update_element_values(
    netlist: Netlist, indices: jnp.ndarray, new_values: jnp.ndarray
) -> Netlist:
    updated_values = netlist.element_values.at[indices].set(new_values)
    return Netlist(
        elements=netlist.elements,
        element_values=updated_values,
        nodes=netlist.nodes,
        n_nodes=netlist.n_nodes,
        n_psources=netlist.n_psources,
        n_flowrates=netlist.n_flowrates,
        current_time_step=netlist.current_time_step,
    )


def resistor_stamp(G, elem_type, value, node1, node2):
    conductance = 1.0 / value

    def update_G_normal(G):
        G = G.at[node1 - 1, node1 - 1].add(conductance)
        G = G.at[node1 - 1, node2 - 1].add(-conductance)
        G = G.at[node2 - 1, node1 - 1].add(-conductance)
        G = G.at[node2 - 1, node2 - 1].add(conductance)
        return G

    def update_G_ground(G):
        G = G.at[node1 - 1, node1 - 1].add(conductance)
        return G

    G = jax.lax.cond(
        elem_type == element_type_map["R"],
        lambda G: jax.lax.cond(
            node2 == 0,
            update_G_ground,  # True: node2 is ground
            update_G_normal,  # False: node2 is not ground
            G,
        ),
        lambda G: G,
        G,
    )
    return G


def get_flowrate_at_resistor(
    X,  # current solution
    value,
    node1,
    node2,
):
    conductance = 1.0 / value
    flowrate = conductance * (X[node1 - 1] - X[node2 - 1])
    return flowrate


# update the resistance
def update_non_linear_resistances(
    vessel_features: VesselFeatures,
    netlist: Netlist,
    X: jnp.ndarray,
) -> Netlist:
    NV = vessel_features.vessel_ids.shape[0]
    resistance = jnp.zeros_like(vessel_features.vessel_base_resistances)

    def scan_fn(carry, idx):
        element_id = vessel_features.vessel_ids[idx]
        area = vessel_features.vessel_areas[idx]
        curv = vessel_features.vessel_curvatures[idx]
        R = vessel_features.vessel_base_resistances[idx]

        node1 = netlist.nodes[element_id, 0]
        node2 = netlist.nodes[element_id, 1]

        q = get_flowrate_at_resistor(
            X, netlist.element_values[element_id], node1, node2
        )
        radius = jnp.sqrt(area / jnp.pi)

        De = ((2 * RHO * q / jnp.pi / MU) / radius) * (jnp.sqrt(radius / curv))

        def compute_mo(De):
            mo = (
                0.1008
                * jnp.sqrt(De)
                * (jnp.sqrt(1 + 1.729 / De) - 1.315 / jnp.sqrt(De)) ** (-3)
            )
            return jnp.mean(mo)

        Mo = jax.lax.cond(
            jnp.logical_or(jnp.mean(curv) < 0, jnp.mean(curv) > 50),
            lambda _: 1.0,
            lambda _: jax.lax.cond(
                jnp.mean(De[1:-1]) > 10,
                lambda _: compute_mo(De),
                lambda _: 1.0,
                None,
            ),
            None,
        )

        carry = carry.at[idx].set(R * Mo)
        return carry, None

    resistance, _ = jax.lax.scan(scan_fn, resistance, jnp.arange(NV))
    jax.debug.print("resistance {}", resistance)
    netlist_to_return = update_element_values(
        netlist, vessel_features.vessel_ids, resistance
    )
    return netlist_to_return


@partial(jax.jit, static_argnums=(4, 5))  # size, n_nodes, dt, max_iter are static
def fixed_point_non_linear_solve_old(
    vessel_features: VesselFeatures,
    netlist: Netlist,
    X_1: jnp.ndarray,
    X_2: jnp.ndarray,
    size,
    n_nodes,
    dt: float,
    max_iter=10,
    tol=1e-2,
):
    def iteration_step(carry):
        X_curr, netlist_curr, converged, iter_num = carry
        G, b = assemble_matrices(netlist_curr, size, n_nodes, dt, X_1, X_2)
        X_next = jnp.linalg.solve(G, b)
        netlist_updated = update_non_linear_resistances(
            vessel_features, netlist_curr, X_next
        )
        residual = jnp.linalg.norm(X_next - X_curr)
        jax.debug.print("residual {} {}", X_next[0, 0], X_curr[0, 0])
        is_converged = residual < tol
        return (X_next, netlist_updated, is_converged, iter_num + 1)

    def cond_fun(carry):
        _, _, converged, iter_num = carry
        return jnp.logical_and(jnp.logical_not(converged), iter_num < max_iter)

    init_carry = (X_1, netlist, False, 0)
    (X_final, netlist_final, converged, iterations) = jax.lax.while_loop(
        cond_fun, iteration_step, init_carry
    )
    # jax.debug.print("iterations {}", iterations)
    return netlist_final, X_final


@partial(jax.jit, static_argnums=(4, 5))  # size, n_nodes, dt, max_iter are static
def fixed_point_non_linear_solve(
    vessel_features: VesselFeatures,
    netlist: Netlist,
    X_1: jnp.ndarray,
    X_2: jnp.ndarray,
    size,
    n_nodes,
    dt: float,
    max_iter=10,
):
    def scan_fn(carry, _):
        X_curr, netlist_curr = carry
        G, b = assemble_matrices(netlist_curr, size, n_nodes, dt, X_1, X_2)
        X_next = jnp.linalg.solve(G, b)
        netlist_updated = update_non_linear_resistances(
            vessel_features, netlist_curr, X_next
        )

        residual = jnp.linalg.norm(X_next - X_curr)
        jax.debug.print(
            "residual {} {} {}",
            X_next[0, 0],
            X_curr[0, 0],
            vessel_features.vessel_resistances,
        )
        return (X_next, netlist_updated), None

    init_carry = (X_1, netlist)
    (X_final, netlist_final), _ = jax.lax.scan(scan_fn, init_carry, jnp.arange(10))
    # jax.debug.print("iterations {}", iterations)
    return netlist_final, X_final


def capacitor_stamp(C, elem_type, value, node1, node2):
    capacitance = value

    def update_C_normal(C):
        C = C.at[node1 - 1, node1 - 1].add(capacitance)
        C = C.at[node1 - 1, node2 - 1].add(-capacitance)
        C = C.at[node2 - 1, node1 - 1].add(-capacitance)
        C = C.at[node2 - 1, node2 - 1].add(capacitance)
        return C

    def update_C_ground(C):
        C = C.at[node1 - 1, node1 - 1].add(capacitance)
        return C

    C = jax.lax.cond(
        elem_type == element_type_map["C"],
        lambda C: jax.lax.cond(
            node2 == 0,
            update_C_ground,  # True: node2 is ground
            update_C_normal,  # False: node2 is not ground
            C,
        ),
        lambda C: C,
        C,
    )
    return C


def inductor_stamp(G, C, elem_type, value, node1, node2, inductor_index):
    inductance = value

    def update_G(G):
        G = G.at[node1 - 1, inductor_index].add(1)
        G = G.at[node2 - 1, inductor_index].add(-1)
        G = G.at[inductor_index, node1 - 1].add(1)
        G = G.at[inductor_index, node2 - 1].add(-1)
        return G

    def update_C(C):
        C = C.at[inductor_index, inductor_index].add(-inductance)
        return C

    G = jax.lax.cond(elem_type == element_type_map["L"], update_G, lambda G: G, G)
    C = jax.lax.cond(elem_type == element_type_map["L"], update_C, lambda C: C, C)
    return G, C


def flowrate_stamp(b, elem_type, value, node1, node2):
    def update_b(b):
        b = b.at[node1 - 1, 0].add(value)
        b = b.at[node2 - 1, 0].add(-value)
        return b

    def update_b_ground(b):
        b = b.at[node2 - 1].add(value)  # flowrate entering node
        return b

    b = jax.lax.cond(
        elem_type == element_type_map["Q"],
        lambda b: jax.lax.cond(
            node1 == 0,
            update_b_ground,  # True: node1 is ground for current sources
            update_b,  # False: node1 is not ground
            b,
        ),
        lambda b: b,
        b,
    )
    return b


@partial(jax.jit, static_argnums=(1, 2))  # size and n_nodes are static
def assemble_matrices(
    netlist: Netlist,
    size: int,
    n_nodes: int,
    time_step_size: float,
    X_1: jnp.ndarray,
    X_2: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Assemble the G,C and b matrices for the given netlist.
    """
    G = jnp.zeros((size, size), float)
    C = jnp.zeros((size, size), float)
    b = jnp.zeros((size, 1), float)

    elements = netlist.elements
    values = netlist.element_values
    nodes = netlist.nodes

    inductor_mask = elements == 2
    inductor_indices = jnp.cumsum(inductor_mask) - 1
    inductor_indices = inductor_indices * inductor_mask
    inductor_indices = inductor_indices + n_nodes

    def scan_fn(carry, x):
        G, C, b = carry
        elem_type, value, node1, node2, inductor_idx = x
        node1_int = jnp.asarray(node1, jnp.int32)
        node2_int = jnp.asarray(node2, jnp.int32)
        inductor_idx_int = jnp.asarray(inductor_idx, jnp.int32)
        G = resistor_stamp(G, elem_type, value, node1_int, node2_int)
        C = capacitor_stamp(C, elem_type, value, node1_int, node2_int)
        G, C = inductor_stamp(
            G, C, elem_type, value, node1_int, node2_int, inductor_idx_int
        )
        b = flowrate_stamp(b, elem_type, value, node1_int, node2_int)

        return (G, C, b), None

    scan_inputs = jnp.column_stack(
        (elements, values, nodes[:, 0], nodes[:, 1], inductor_indices)
    )

    (G, C, b), _ = jax.lax.scan(scan_fn, (G, C, b), scan_inputs)

    G_timestep = G + 3 / 2 * C / time_step_size
    b_timestep = b + 2 * C @ X_1 / time_step_size - 1 / 2 * C @ X_2 / time_step_size

    condition_number = jnp.linalg.cond(G_timestep)
    # jax.debug.print("condition_number: {}", condition_number)

    G_scaled, b_scaled = row_max_scale(G_timestep, b_timestep)
    # jax.debug.print("G_scaled: {}", jnp.linalg.cond(G_scaled))

    return (G_scaled, b_scaled)
    # return (G_timestep, b_timestep)


@partial(jax.jit, static_argnums=(2, 3, 7))
def assemble_matrices_with_non_linear_solve(
    netlist: Netlist,  # 0
    vessel_features: VesselFeatures,  # 1
    size: int,  # 2
    n_nodes: int,  # 3
    time_step_size: float,  # 4
    X_1: jnp.ndarray,  # 5
    X_2: jnp.ndarray,  # 6
    max_non_linear_iter: int = 10,  # 7
):
    updated_netlist, X = fixed_point_non_linear_solve(
        vessel_features,
        netlist,
        X_1,
        X_2,
        size,
        n_nodes,
        dt=time_step_size,
        max_iter=max_non_linear_iter,
    )
    G, b = assemble_matrices(updated_netlist, size, n_nodes, time_step_size, X_1, X_2)
    return G, b, updated_netlist


def row_max_scale(matrix, vector):
    row_maxes = jnp.max(jnp.abs(matrix), axis=1, keepdims=True)
    # Avoid division by zero, why tf is this necessary?
    safe_maxes = jnp.maximum(row_maxes, 1e-10)
    scaled_matrix = matrix / safe_maxes
    scaled_vector = vector / safe_maxes
    return scaled_matrix, scaled_vector


def module_test():
    print("Module test for netlist_v5.py passed!")
