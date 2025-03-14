import jax
import json
from functools import partial
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, NamedTuple, Tuple

# GLOBALS
DEBUG = False

element_type_map = {
    "R": 0,  # resistor
    "C": 1,  # capacitor
    "L": 2,  # inductor
    "Q": 3,  # flowrate source
    "P": 4,  # pressure source
    "Rl": 5,  # non-linear resistor
}

MU = 0.04
RHO = 1.06
LARGE_NUMBER = 1e6
SMALL_NUMBER = 1e-10

# file paths
output_path = "./output/"


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


# TODO: implement boolean flag for parameters that need to be optimized from within the json file itself.
def read_base_elements(file_path: str) -> List[Dict]:
    with open(file_path, "r") as f:
        data = json.load(f)
        base_elements = data["elements"]
    return base_elements


# to ensure R is always followed by L, needed for querying flowrate for non linear resistance
def reorganize_elements(base_elements: List[Dict]) -> List[Dict]:
    if DEBUG:
        json.dump(
            base_elements, open(output_path + "elements_base.json", "w"), indent=4
        )

    rl_elements = [elem for elem in base_elements if elem["type"] == "Rl"]
    l_elements = [elem for elem in base_elements if elem["type"] == "L"]

    rl_output_to_elem = {rl["node2"]: rl for rl in rl_elements}

    # Map inductors to their corresponding resistors
    l_to_rl_map = {}
    for l_elem in l_elements:
        input_node = l_elem["node1"]
        if input_node in rl_output_to_elem:
            l_to_rl_map[l_elem["name"]] = rl_output_to_elem[input_node]

    new_elements = []
    processed_rl_names = set()

    source_elements = [elem for elem in base_elements if elem["type"] in ["Q", "P"]]
    new_elements.extend(source_elements)

    for rl in rl_elements:
        if rl["name"] not in processed_rl_names:
            new_elements.append(rl)
            processed_rl_names.add(rl["name"])
            for l_elem in l_elements:
                if l_elem["node1"] == rl["node2"]:
                    new_elements.append(l_elem)
                    break

    remaining_elements = [elem for elem in base_elements if elem not in new_elements]
    new_elements.extend(remaining_elements)

    if DEBUG:
        json.dump(new_elements, open(output_path + "elements_new.json", "w"), indent=4)

    return new_elements


def create_netlist(file_path: str) -> Netlist:
    type_map = element_type_map
    base_elements = read_base_elements(file_path)
    base_elements = reorganize_elements(base_elements)
    elements = jnp.array([type_map[e["type"]] for e in base_elements])
    element_values = jnp.array([e["value"] for e in base_elements])
    nodes = jnp.array([[e["node1"], e["node2"]] for e in base_elements], int)

    if DEBUG:
        print(f"elements: {elements}")
        print(f"element_values: {element_values}")
        print(f"nodes: {nodes}")

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


def update_vessel_features(vessel_features, new_resistances):
    return VesselFeatures(
        vessel_ids=vessel_features.vessel_ids,
        vessel_areas=vessel_features.vessel_areas,
        vessel_curvatures=vessel_features.vessel_curvatures,
        vessel_lengths=vessel_features.vessel_lengths,
        vessel_base_resistances=vessel_features.vessel_base_resistances,
        vessel_resistances=new_resistances,
    )


def resistor_stamp(
    G: jnp.ndarray,
    elem_type: int,
    value: float,
    node1: jnp.ndarray,
    node2: jnp.ndarray,
):
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


def non_linear_resistor_stamp(
    G: jnp.ndarray,
    b: jnp.ndarray,
    elem_type: int,
    value: float,
    S: float,
    node1: jnp.ndarray,
    node2: jnp.ndarray,
    q_prev_at_node: float,
):
    q_prev = q_prev_at_node
    Rb = value

    R_eq = Rb + 2 * S * q_prev
    P_eq = -S * q_prev**2
    conductance = 1.0 / R_eq

    def update_G_normal(G):
        G = G.at[node1 - 1, node1 - 1].add(conductance)
        G = G.at[node1 - 1, node2 - 1].add(-conductance)
        G = G.at[node2 - 1, node1 - 1].add(-conductance)
        G = G.at[node2 - 1, node2 - 1].add(conductance)
        return G

    def update_b_normal(b):
        b = b.at[node1 - 1, 0].add(conductance * P_eq)
        b = b.at[node2 - 1, 0].add(-conductance * P_eq)
        return b

    def update_G_ground(G):
        G = G.at[node1 - 1, node1 - 1].add(conductance)
        return G

    def update_b_ground(b):
        b = b.at[node1 - 1, 0].add(conductance * P_eq)
        return b

    G = jax.lax.cond(
        jnp.logical_and(elem_type == element_type_map["Rl"], q_prev != -LARGE_NUMBER),
        lambda G: jax.lax.cond(
            node2 == 0,
            update_G_ground,  # If node2 is ground
            update_G_normal,  # If both nodes are not ground
            G,
        ),
        lambda G: G,
        G,
    )

    b = jax.lax.cond(
        jnp.logical_and(elem_type == element_type_map["Rl"], q_prev != -LARGE_NUMBER),
        lambda b: jax.lax.cond(
            node2 == 0,
            update_b_ground,  # If node2 is ground
            update_b_normal,  # If both nodes are not ground
            b,
        ),
        lambda b: b,
        b,
    )

    return G, b


def create_inductor_index_mask(netlist: Netlist):
    n_elements = len(netlist.elements)
    mask = np.full((n_elements,), -1, dtype=np.int32)
    inductor_start_idx = netlist.n_nodes + netlist.n_psources
    inductor_count = 0

    for i in range(len(netlist.elements)):
        if netlist.elements[i] == element_type_map["L"]:
            # inductor's index in the solution vector
            sol_idx = inductor_start_idx + inductor_count
            inductor_count += 1
            mask[i] = sol_idx
        else:
            mask[i] = -1

    return jnp.array(mask, dtype=jnp.int32)


def create_flowrate_query_indices(inductor_index_mask: jnp.ndarray):
    mask = inductor_index_mask != -1
    flowrate_query_indices = jnp.full_like(inductor_index_mask, -1)
    flowrate_query_indices = flowrate_query_indices.at[:-1].set(
        jnp.where(mask[1:], inductor_index_mask[1:], flowrate_query_indices[:-1])
    )
    flowrate_query_indices = flowrate_query_indices.at[0].set(
        jnp.where(mask[0], inductor_index_mask[0], -1)
    )
    return flowrate_query_indices


def get_flowrate_at_resistor(
    X: jnp.ndarray,
    value: float,
    node1: int,
    node2: int,
):
    conductance = 1.0 / value
    flowrate = jax.lax.cond(
        node2 == 0,
        lambda _: conductance * X[node1 - 1],  # ground resistor
        lambda _: conductance * (X[node1 - 1] - X[node2 - 1]),  # internal resistor
        None,
    )
    return flowrate


def capacitor_stamp(
    C: jnp.ndarray,
    elem_type: int,
    value: float,
    node1: jnp.ndarray,
    node2: jnp.ndarray,
):
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


def inductor_stamp(
    G: jnp.ndarray,
    C: jnp.ndarray,
    elem_type: int,
    value: float,
    node1: jnp.ndarray,
    node2: jnp.ndarray,
    inductor_index: jnp.ndarray,
):
    inductance = value
    inductor_index = inductor_index.astype(jnp.int32)

    def update_G_normal(G):
        G = G.at[node1 - 1, inductor_index].add(1)
        G = G.at[node2 - 1, inductor_index].add(-1)
        G = G.at[inductor_index, node1 - 1].add(1)
        G = G.at[inductor_index, node2 - 1].add(-1)
        return G

    def update_C_normal(C):
        C = C.at[inductor_index, inductor_index].add(-inductance)
        return C

    def update_G_ground(G):
        G = G.at[node1 - 1, inductor_index].add(1)
        G = G.at[inductor_index, node1 - 1].add(1)
        return G

    def update_C_ground(C):
        C = C.at[inductor_index, inductor_index].add(-inductance)
        return C

    G = jax.lax.cond(
        elem_type == element_type_map["L"],
        lambda G: jax.lax.cond(
            node2 == 0,
            update_G_ground,
            update_G_normal,
            G,
        ),
        lambda G: G,
        G,
    )

    C = jax.lax.cond(
        elem_type == element_type_map["L"],
        lambda C: jax.lax.cond(
            node2 == 0,
            update_C_ground,
            update_C_normal,
            C,
        ),
        lambda C: C,
        C,
    )

    return G, C


def flowrate_stamp(
    b: jnp.ndarray,
    elem_type: int,
    value: float,
    node1: jnp.ndarray,
    node2: jnp.ndarray,
):
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


@partial(jax.jit, static_argnums=(1))
def assemble_matrices(
    netlist: Netlist,
    size: int,  # static
    inductor_indices: jnp.ndarray,
    flowrate_query_indices: jnp.ndarray,
    time_step_size: float,
    X_1: jnp.ndarray,
    X_2: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    G = jnp.zeros((size, size), float)
    C = jnp.zeros((size, size), float)
    b = jnp.zeros((size, 1), float)

    elements = netlist.elements
    values = netlist.element_values
    nodes = netlist.nodes

    q_prev = jnp.full_like(inductor_indices, -LARGE_NUMBER, float)

    mask = flowrate_query_indices != -1

    jax.lax.cond(
        DEBUG,
        lambda _: jax.debug.print(
            "mask: {}",
            mask,
        ),
        lambda _: None,
        None,
    )

    # avoid indexing out of bounds for X_1
    safe_indices = jnp.maximum(0, flowrate_query_indices)

    jax.lax.cond(
        DEBUG,
        lambda _: jax.debug.print(
            "safe indices: {}",
            safe_indices,
        ),
        lambda _: None,
        None,
    )

    # NOTE: Advanced indexing using gather operation
    # jnp.arange(q_prev.shape[0]) generates indices [0, 1, 2, ..., q_prev.shape[0]-1]
    # X_1[safe_indices, 0] gathers values from X_1 at the indices specified by safe_indices
    q_prev = q_prev.at[jnp.arange(q_prev.shape[0])].set(
        jnp.where(mask, X_1[safe_indices, 0], q_prev)
    )

    S_array = jnp.zeros_like(q_prev)
    S_array = jnp.where(q_prev != -LARGE_NUMBER, 0.1, S_array)

    jax.lax.cond(
        DEBUG,
        lambda _: jax.debug.print(
            "inductor_indices: {} {}",
            flowrate_query_indices,
            q_prev,
        ),
        lambda _: None,
        None,
    )

    # scan over netlist elements
    def scan_fn(carry, x):
        G, C, b = carry
        elem_type, value, node1, node2, inductor_idx, q_prev_query, S = x
        node1_int = jnp.asarray(node1, jnp.int32)
        node2_int = jnp.asarray(node2, jnp.int32)
        G = resistor_stamp(G, elem_type, value, node1_int, node2_int)
        G, b = non_linear_resistor_stamp(
            G, b, elem_type, value, S, node1_int, node2_int, q_prev_query
        )
        C = capacitor_stamp(C, elem_type, value, node1_int, node2_int)
        G, C = inductor_stamp(
            G, C, elem_type, value, node1_int, node2_int, inductor_idx
        )
        b = flowrate_stamp(b, elem_type, value, node1_int, node2_int)

        return (G, C, b), None

    scan_inputs = jnp.column_stack(
        (elements, values, nodes[:, 0], nodes[:, 1], inductor_indices, q_prev, S_array)
    )

    (G, C, b), _ = jax.lax.scan(scan_fn, (G, C, b), scan_inputs)

    # BDF2 time stepping
    G_timestep = G + 3 / 2 * C / time_step_size
    b_timestep = b + 2 * C @ X_1 / time_step_size - 1 / 2 * C @ X_2 / time_step_size

    condition_number = jnp.linalg.cond(G_timestep)

    jax.lax.cond(
        DEBUG,
        lambda _: jax.debug.print("condition_number: {}", condition_number),
        lambda _: None,
        None,
    )

    G_scaled, b_scaled = row_max_scale(G_timestep, b_timestep)

    jax.lax.cond(
        DEBUG,
        lambda _: jax.debug.print(
            "condition number of scaled matrix: {}", jnp.linalg.cond(G_scaled)
        ),
        lambda _: None,
        None,
    )
    return (G_scaled, b_scaled)


def row_max_scale(matrix, vector):
    row_maxes = jnp.max(jnp.abs(matrix), axis=1, keepdims=True)
    safe_maxes = jnp.maximum(row_maxes, 1e-10)
    scaled_matrix = matrix / safe_maxes
    scaled_vector = vector / safe_maxes
    return scaled_matrix, scaled_vector
