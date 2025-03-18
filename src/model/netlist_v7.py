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


class JunctionFeatures(NamedTuple):
    junction_resistor_ids: jnp.ndarray  # (NJ X 3)
    junction_areas: jnp.ndarray  # (NJ X 3)
    junction_angles: jnp.ndarray  # (NJ X 3)


def create_junction_features(file_path: str) -> JunctionFeatures:
    with open(file_path, "r") as f:
        data = json.load(f)

    junctions = data.get("junctions", [])
    num_junctions = len(junctions)

    resistor_ids = np.zeros((num_junctions, 3), dtype=np.int32)
    areas = np.zeros((num_junctions, 3), dtype=np.float32)
    angles = np.zeros((num_junctions, 3), dtype=np.float32)

    for i, junction in enumerate(junctions):
        resistor_ids[i] = junction["resistor_ids"]
        areas[i] = junction["areas"]

        angle_values = junction["angles"]
        angles[i] = angle_values

    junction_resistor_ids = jnp.array(resistor_ids, int)
    junction_areas = jnp.array(areas, float)
    junction_angles = jnp.array(angles, float)

    return JunctionFeatures(
        junction_resistor_ids=junction_resistor_ids,
        junction_areas=junction_areas,
        junction_angles=junction_angles,
    )


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


# to ensure R is always followed by L, needed for querying  previous timestep flowrate for non linearized resistance
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
    # base_elements = reorganize_elements(base_elements)
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


# non linear contribution from stenosis, linearized around q_prev
def linearized_resistor_stamp(
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


# stores the index of the inductor curents in the solution vector in the position of the inductor element in the netlist
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


# shifts the inductor indices by one to couple the inductor current to the previous coupled element, a resistor.
# reorg netlist ensures that the order of netlist elements is correct, so we only need to shift the indices by one.
# used to query the flowrate at the inductor's coupled resistor for linearized resistance for the bvessel
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


def modification_factor_update(
    vessel_features: VesselFeatures,
    netlist: Netlist,
    X: jnp.ndarray,
) -> Netlist:
    NV = vessel_features.vessel_ids.shape[0]
    vessel_resistances = jnp.zeros_like(vessel_features.vessel_base_resistances)

    # scans through all the vessels
    def scan_fn(carry, idx):
        element_id = vessel_features.vessel_ids[idx]
        area = vessel_features.vessel_areas[idx]
        curv = vessel_features.vessel_curvatures[idx]
        R_base = vessel_features.vessel_base_resistances[idx]

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
                jnp.mean(De) > 10,
                lambda _: compute_mo(De),
                lambda _: 1.0,
                None,
            ),
            None,
        )
        R_new = R_base * Mo
        carry = carry.at[idx].set(R_new)
        return carry, None

    vessel_resistances_updated, _ = jax.lax.scan(
        scan_fn, vessel_resistances, jnp.arange(NV)
    )

    netlist_to_return = update_element_values(
        netlist, vessel_features.vessel_ids, vessel_resistances_updated
    )

    _ = jax.lax.cond(
        DEBUG,
        lambda _: jax.debug.print(
            "updated netlist values : {} vs older netlist vlaues : {}",
            netlist_to_return.element_values,
            netlist.element_values,
        ),
        lambda _: None,
        None,
    )

    return netlist_to_return


@jax.jit
def junction_loss_update(
    junction_features: JunctionFeatures,
    netlist: Netlist,
    X: jnp.ndarray,
):
    def process_junction(netlist_acc, j_idx):
        r_ids = junction_features.junction_resistor_ids[j_idx]

        def scan_branches(carry, branch_idx):
            r_id = r_ids[branch_idx]
            # jax.debug.print("branch idx: {}, resistor id: {}", branch_idx, r_id)

            node1 = netlist_acc.nodes[r_id, 0]
            node2 = netlist_acc.nodes[r_id, 1]
            # jax.debug.print("resistor id: {}, node1: {}, node2: {}", r_id, node1, node2)

            q = get_flowrate_at_resistor(
                X, netlist_acc.element_values[r_id], node1, node2
            )
            flowrates, _ = carry
            new_flowrates = flowrates.at[branch_idx].set(q[0])
            return (new_flowrates, None), None

        init_flowrates = jnp.zeros(3)

        (junction_flowrates, _), _ = jax.lax.scan(
            scan_branches, (init_flowrates, None), jnp.arange(3)
        )

        junction_areas = junction_features.junction_areas[j_idx].reshape(3, 1)
        junction_angles = junction_features.junction_angles[j_idx].reshape(3, 1)
        junction_angles = jnp.deg2rad(junction_angles)
        junction_flowrates = (junction_flowrates).reshape(3, 1)
        junction_velocities = junction_flowrates / junction_areas
        junction_velocities = junction_velocities.at[1:, 0].set(
            -junction_velocities[1:, 0]
        )

        # jax.debug.print(
        #     "\njunction velocities: {},\njunction_areas: {},\njunction angles: {}\n",
        #     junction_velocities,
        #     junction_flowrates,
        #     junction_angles,
        # )
        Ucom, K = junction_loss_coefficient(
            junction_velocities, junction_areas, junction_angles
        )
        Ucom = Ucom[0, 0]
        # jax.debug.print("junction loss coefficient: {}, {}", Ucom, K)
        K_vals = jnp.array([0.0, K[0, 0], K[1, 0]]).reshape(3, 1)
        # jax.debug.print("K_vals: {}", K_vals)
        base_resistances = netlist_acc.element_values[r_ids]
        bif_dis = (
            0.5
            * RHO
            * Ucom**2
            * K_vals
            / (junction_flowrates + jnp.full_like(junction_flowrates, SMALL_NUMBER))
        ).reshape(
            3,
        )
        # jax.debug.print("bif_dis: {}", base_resistances.shape)
        new_resistances = base_resistances * (1.0 + bif_dis)
        netlist_acc = update_element_values(netlist_acc, r_ids, new_resistances)

        return netlist_acc, None

    num_junctions = junction_features.junction_resistor_ids.shape[0]
    updated_netlist, _ = jax.lax.scan(
        process_junction, netlist, jnp.arange(num_junctions)
    )

    return updated_netlist


def get_flowrate_at_resistor(
    X: jnp.ndarray,
    value,
    node1: jnp.ndarray,
    node2: jnp.ndarray,
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
    S_array = jnp.where(q_prev != -LARGE_NUMBER, 0.001, S_array)

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
        G, b = linearized_resistor_stamp(
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


@partial(jax.jit, static_argnums=(3))  # size is static
def fixed_point_non_linear_solve(
    vessel_features: VesselFeatures,
    junction_features: JunctionFeatures,
    netlist: Netlist,
    size: int,
    inductor_indices: jnp.ndarray,
    flowrate_query_indices: jnp.ndarray,
    X_1: jnp.ndarray,
    X_2: jnp.ndarray,
    dt: float,
):
    def scan_fn(carry, _):
        _, netlist_curr = carry
        netlist_curr = junction_loss_update(junction_features, netlist_curr, X_1)
        G, b = assemble_matrices(
            netlist_curr,
            size,
            inductor_indices,
            flowrate_query_indices,
            dt,
            X_1,
            X_2,
        )

        X_next = jnp.linalg.solve(G, b)
        netlist_updated = modification_factor_update(
            vessel_features, netlist_curr, X_next
        )
        return (X_next, netlist_updated), None

    init_carry = (X_1, netlist)

    # NOTE: Cannot pass max-iter in the arange iteration arr for scan as a function arg due to concreteness issues with JAX
    # hard coded for now, might change in the future
    (X_final, netlist_final), _ = jax.lax.scan(scan_fn, init_carry, jnp.arange(10))

    return netlist_final, X_final


@partial(jax.jit, static_argnums=(3))
def assemble_matrices_with_non_linear_solve(
    netlist: Netlist,  # 0
    vessel_features: VesselFeatures,  # 1
    junction_features: JunctionFeatures,  # 2
    size: int,  # 2
    inductor_indices: jnp.ndarray,  # 3
    flowrate_query_indices: jnp.ndarray,  # 4
    time_step_size: float,  # 4
    X_1: jnp.ndarray,  # 5
    X_2: jnp.ndarray,  # 6
):
    updated_netlist, X = fixed_point_non_linear_solve(
        vessel_features,
        junction_features,
        netlist,
        size,
        inductor_indices,
        flowrate_query_indices,
        X_1,
        X_2,
        dt=time_step_size,
    )
    G, b = assemble_matrices(
        netlist,
        size,
        inductor_indices,
        flowrate_query_indices,
        time_step_size,
        X_1,
        X_2,
    )

    return G, b, updated_netlist


@jax.jit
def junction_loss_coefficient(U: jnp.ndarray, A: jnp.ndarray, theta: jnp.ndarray):
    xwrap = jnp.remainder(theta, 2 * jnp.pi)
    mask = jnp.abs(xwrap) > jnp.pi
    correction = 2 * jnp.pi * jnp.sign(xwrap)
    xwrap = jnp.where(mask, xwrap - correction, xwrap)
    theta = xwrap

    def zero_flow_case(U, A, theta):
        return jnp.array([[0.0]]), jnp.zeros((2, 1), float)

    def converging_flow(
        U: jnp.ndarray, A: jnp.ndarray, theta: jnp.ndarray, Q: jnp.ndarray
    ):
        Q_si = Q[0:2,]
        Q_ci = Q[2:3,]
        U_si = U[0:2,]
        U_ci = U[2:3,]
        Qtot = jnp.sum(Q[0:2,])
        FlowRatio = -Q_ci / Qtot

        theta_si = theta[0:2,]
        theta_ci = theta[2:3,]
        PseudoColAngle = jnp.mean(theta_ci)
        sin_weighted = jnp.sum(jnp.sin(theta_si) * Q_si)
        cos_weighted = jnp.sum(jnp.cos(theta_si) * Q_si)
        PseudoSupAngle = jnp.arctan2(sin_weighted, cos_weighted)
        PseudoColAngle = jax.lax.cond(
            jnp.abs(PseudoSupAngle - PseudoColAngle) < 0.5 * jnp.pi,
            lambda x: x + jnp.pi,
            lambda x: x,
            PseudoColAngle,
        )
        theta = wrapTopi(theta - PseudoColAngle)
        theta_si = theta[0:2,]
        theta_ci = theta[2:3,]
        pseudodirection = jnp.sign(jnp.mean(jnp.sin(theta_si) * Q_si))
        theta = jax.lax.cond(pseudodirection < 0, lambda x: -x, lambda x: x, theta)
        theta_si = theta[0:2,]
        theta_ci = theta[2:3,]
        sin_weighted_abs = jnp.sum(jnp.sin(jnp.abs(theta_si)) * Q_si)
        cos_weighted_abs = jnp.sum(jnp.cos(jnp.abs(theta_si)) * Q_si)
        PseudoSupAngle = jnp.arctan2(sin_weighted_abs, cos_weighted_abs)
        etransferfactor = (
            0.8 * (jnp.pi - PseudoSupAngle) * jnp.sign(theta_ci) - 0.2
        ) * (1 - FlowRatio)
        U_Q_weighted = jnp.sum(U_si * Q_si) / Qtot
        TotPseudoArea = Qtot / ((1 - etransferfactor) * U_Q_weighted)
        A_ci = A[2:3,]
        AreaRatio = TotPseudoArea / (A_ci)
        theta = wrapTo2pi(PseudoSupAngle - theta_ci)
        phi = theta

        C_val = (1 - jnp.exp(-FlowRatio / 0.02)) * (
            1 - (1.0 / (AreaRatio * FlowRatio)) * jnp.cos(0.75 * (jnp.pi - phi))
        )
        Ucom = U_ci
        K = (U_ci**2 / (Ucom**2)) * (2 * C_val + (U_si**2) / (U_ci**2) - 1)

        jax.lax.cond(
            DEBUG,
            lambda _: jax.debug.print(
                "printing out K from converging boolean branch \n\n{} \n\n{} \n\n {}\n",
                Ucom,
                C_val,
                K,
            ),
            lambda _: None,
            None,
        )

        return Ucom.reshape(1, 1) * 0, K * 0

    def diverging_flow(
        U: jnp.ndarray,
        A: jnp.ndarray,
        theta: jnp.ndarray,
        Q: jnp.ndarray,
    ):
        Q_si = Q[0:1,]
        Q_ci = Q[1:3,]
        U_si = U[0:1,]
        U_ci = U[1:3,]
        Qtot = jnp.sum(Q[0:1,])
        FlowRatio = -Q_ci / Qtot

        theta_si = theta[0:1,]
        theta_ci = theta[1:3,]
        PseudoColAngle = jnp.mean(theta_ci)
        sin_weighted = jnp.sum(jnp.sin(theta_si) * Q_si)
        cos_weighted = jnp.sum(jnp.cos(theta_si) * Q_si)
        PseudoSupAngle = jnp.arctan2(sin_weighted, cos_weighted)
        PseudoColAngle = jax.lax.cond(
            jnp.abs(PseudoSupAngle - PseudoColAngle) < 0.5 * jnp.pi,
            lambda x: x + jnp.pi,
            lambda x: x,
            PseudoColAngle,
        )
        theta = wrapTopi(theta - PseudoColAngle)
        theta_si = theta[0:1,]
        theta_ci = theta[1:3,]
        pseudodirection = jnp.sign(jnp.mean(jnp.sin(theta_si) * Q_si))
        theta = jax.lax.cond(pseudodirection < 0, lambda x: -x, lambda x: x, theta)
        theta_si = theta[0:1,]
        theta_ci = theta[1:3,]

        sin_weighted_abs = jnp.sum(jnp.sin(jnp.abs(theta_si)) * Q_si)
        cos_weighted_abs = jnp.sum(jnp.cos(jnp.abs(theta_si)) * Q_si)
        PseudoSupAngle = jnp.arctan2(sin_weighted_abs, cos_weighted_abs)
        etransferfactor = (
            0.8 * (jnp.pi - PseudoSupAngle) * jnp.sign(theta_ci) - 0.2
        ) * (1 - FlowRatio)
        U_Q_weighted = jnp.sum(U_si * Q_si) / Qtot
        TotPseudoArea = Qtot / ((1 - etransferfactor) * U_Q_weighted)
        A_ci = A[1:3,]
        AreaRatio = TotPseudoArea / (A_ci)
        theta = wrapTo2pi(PseudoSupAngle - theta_ci)
        phi = theta

        C_val = (1 - jnp.exp(-FlowRatio / 0.02)) * (
            1 - (1.0 / (AreaRatio * FlowRatio)) * jnp.cos(0.75 * (jnp.pi - phi))
        )

        Ucom = U_si
        K = (U_ci**2 / (Ucom**2)) * (2 * C_val + (U_si**2) / (U_ci**2) - 1)

        jax.lax.cond(
            DEBUG,
            lambda _: jax.debug.print(
                "printing out K from diverging boolean branch \n\n{} \n\n{} \n\n {}\n",
                Ucom,
                C_val,
                K,
            ),
            lambda _: None,
            None,
        )
        return Ucom.reshape(1, 1), K

    Q = U * A
    Si = Q >= 0.0
    is_zero_flow = jnp.all(jnp.abs(Q) < 1e-7)
    is_converging = jnp.sum(Si) > 1
    # jax.debug.print(
    #     "is_converging: {} \n is_zero_flow: {}", is_converging, is_zero_flow
    # )
    return jax.lax.cond(
        is_zero_flow,
        lambda: zero_flow_case(U, A, theta),
        lambda: jax.lax.cond(
            is_converging,
            lambda: converging_flow(U, A, theta, Q),
            lambda: diverging_flow(U, A, theta, Q),
        ),
    )


def wrapTopi(theta: jnp.ndarray):
    xwrap = jnp.remainder(theta, 2 * jnp.pi)
    mask = jnp.abs(xwrap) > jnp.pi
    correction = 2 * jnp.pi * jnp.sign(xwrap)
    xwrap = jnp.where(mask, xwrap - correction, xwrap)
    theta = xwrap
    return theta


def wrapTo2pi(theta: jnp.ndarray):
    xwrap = jnp.remainder(theta, 4 * jnp.pi)
    mask = jnp.abs(xwrap) > 2 * jnp.pi
    correction = 4 * jnp.pi * jnp.sign(xwrap)
    xwrap = jnp.where(mask, xwrap - correction, xwrap)
    theta = xwrap
    return theta
