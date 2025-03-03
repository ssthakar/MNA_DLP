import jax
import json
from functools import partial
import jax.numpy as jnp
from typing import Dict, List, NamedTuple, Tuple

element_type_map = {
    "R": 0,  # resistor
    "C": 1,  # capacitor
    "L": 2,  # inductor
    "Q": 3,  # flowrate source
    "P": 4,  # pressure source
}


class Netlist(NamedTuple):
    elements: jnp.ndarray
    element_values: jnp.ndarray
    nodes: jnp.ndarray
    n_nodes: int
    n_psources: int
    n_flowrates: int
    current_time_step: int


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

    n_nodes = int(jnp.max(nodes) + 1)

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


def resistor_stamp(
    G,
    C,
    elem_type,
    value,
    node1,
    node2,
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
    return G, C


def capacitor_stamp(
    G,
    C,
    elem_type,
    value,
    node1,
    node2,
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
    return G, C


def inductor_stamp(
    G,
    C,
    elem_type,
    value,
    node1,
    node2,
    inductor_index,
):
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


@partial(jax.jit, static_argnums=(1, 2))
def assemble_matrices(
    netlist: Netlist,
    size: int,
    n_nodes: int,
    time_step_size: float,
    X_prev: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
        G, C = resistor_stamp(G, C, elem_type, value, node1_int, node2_int)
        G, C = capacitor_stamp(G, C, elem_type, value, node1_int, node2_int)
        G, C = inductor_stamp(
            G, C, elem_type, value, node1_int, node2_int, inductor_idx_int
        )
        return (G, C, b), None

    scan_inputs = jnp.column_stack(
        (elements, values, nodes[:, 0], nodes[:, 1], inductor_indices)
    )

    (G, C, b), _ = jax.lax.scan(scan_fn, (G, C, b), scan_inputs)
    G_timestep = G + C / time_step_size

    return (G_timestep, b)


def module_test():
    print("Module test for netlist_v3.py passed!")
