import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple


def create_netlist() -> Tuple[
    List[Tuple[int, int, float, bool]],  # Resistors
    List[Tuple[int, int, float, bool]],  # Capacitors
    List[Tuple[int, int, float, bool]],  # Inductors
    List[Tuple[int, int, jnp.ndarray, bool]],  # Current Sources
]:
    """
    Return an empty netlist as a tuple of lists.
    Each list contains a tuple containing:
        integers: represent nodes.
        floats/jnp.ndarray: represent values/time series values
        boolean: represents if the element is being optimized
    """
    return ([], [], [], [])


def add_component(
    netlist,  # empty netlist
    component_type: str,  # component type
    n1: int,  # source node
    n2: int,  # sink node
    value: float,  # value of the component
    time_series: jnp.ndarray,  # time series for current sources
):
    """
    Adds component to the netlist.
    """
    resistors, capacitors, inductors, current_sources = netlist
    if component_type == "resistors":
        return (resistors + [(n1, n2, value)], capacitors, inductors, current_sources)
    elif component_type == "capacitors":
        return (resistors, capacitors + [(n1, n2, value)], inductors, current_sources)
    elif component_type == "inductors":
        return (resistors, capacitors, inductors + [(n1, n2, value)], current_sources)
    elif component_type == "voltage_sources":
        return (
            resistors,
            capacitors,
            inductors,
            current_sources + [(n1, n2, time_series)],
        )
    else:
        raise ValueError(f"Unknown component type: {component_type}")


def update_resistor(netlist, resistor_idx: int, new_value: float):
    """
    updates the value of a component in the netlist.
    """
    resistors, _, _, _ = netlist
    resistors[resistor_idx] = (
        resistors[resistor_idx][0],
        resistors[resistor_idx][1],
        new_value,
    )
