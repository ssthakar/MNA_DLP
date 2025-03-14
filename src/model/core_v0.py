import jax
import jax.numpy as jnp
from typing import Dict, List, NamedTuple, Tuple
from src.model.io_v0 import read_netlist_json

# thermo-physical properties
physical_props = {
    "mu": 0.04,
    "rho": 1.06,
}


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


def create_netlist(file_path: str) -> Netlist:
    type_map = element_type_map
    base_elements = read_netlist_json(file_path)
    elements = jnp.array([type_map[e["type"]] for e in base_elements])
    element_values = jnp.array([e["value"] for e in base_elements])
    nodes = jnp.array([[e["node1"], e["node2"]] for e in base_elements], int)

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


class VesselFeatures(NamedTuple):
    vessel_ids: jnp.ndarray  # (NV X 1)
    vessel_areas: jnp.ndarray  # (NV X N_seg)
    vessel_curvatures: jnp.ndarray  # (NV X N_seg)
    vessel_lengths: jnp.ndarray  # (NV X N_seg)
    vessel_base_resistances: (
        jnp.ndarray
    )  # (NV X 1) (Poisuille + Stenosis + Curvature effects)
    vessel_resistances: jnp.ndarray  # (NV X 1) # after non-linear update


def update_vessel_features(vessel_features, new_resistances):
    return VesselFeatures(
        vessel_ids=vessel_features.vessel_ids,
        vessel_areas=vessel_features.vessel_areas,
        vessel_curvatures=vessel_features.vessel_curvatures,
        vessel_lengths=vessel_features.vessel_lengths,
        vessel_base_resistances=vessel_features.vessel_base_resistances,
        vessel_resistances=new_resistances,
    )
