import jax
import jax.numpy as jnp
import src.model.netlist_v7 as netlist

RHO = 1.06
MU = 0.04


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


@jax.jit
def junction_loss_coefficient(U: jnp.ndarray, A: jnp.ndarray, theta: jnp.ndarray):
    """
    Function to compute junction loss at a three way junction.
    Converging/Diverging flow depends on network and topology

    Args:
        U : Flow rate at the three nodes that make the junction.
        A : Cross sectional areas of junction segments
        theta : The angle the daughter segments make with the supplier segment.
    Returns:
        Ucom,K
    """
    xwrap = jnp.remainder(theta, 2 * jnp.pi)
    mask = jnp.abs(xwrap) > jnp.pi
    correction = 2 * jnp.pi * jnp.sign(xwrap)
    xwrap = jnp.where(mask, xwrap - correction, xwrap)
    theta = xwrap

    # jax.debug.print("printing out U {}", U)

    # zero flow rate at node
    def zero_flow_case(U, A, theta):
        return jnp.array([[0.0]]), jnp.zeros((2, 1), float)

    # single inlet into two outlets
    def converging_flow(
        U: jnp.ndarray, A: jnp.ndarray, theta: jnp.ndarray, Q: jnp.ndarray
    ):
        # get supplier and collector segments from the function args
        Q_si = Q[0:2,]
        Q_ci = Q[2:3,]
        U_si = U[0:2,]
        U_ci = U[2:3,]
        # get total flow rate and flow ratio
        Qtot = jnp.sum(Q[0:2,])
        FlowRatio = -Q_ci / Qtot

        # get datum angle
        theta_si = theta[0:2,]
        theta_ci = theta[2:3,]
        # Reorient all branch angles so average collector angle is 0
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
        # calculate pseudosupplier angle
        pseudodirection = jnp.sign(jnp.mean(jnp.sin(theta_si) * Q_si))
        theta = jax.lax.cond(pseudodirection < 0, lambda x: -x, lambda x: x, theta)
        theta_si = theta[0:2,]
        theta_ci = theta[2:3,]

        sin_weighted_abs = jnp.sum(jnp.sin(jnp.abs(theta_si)) * Q_si)
        cos_weighted_abs = jnp.sum(jnp.cos(jnp.abs(theta_si)) * Q_si)
        PseudoSupAngle = jnp.arctan2(sin_weighted_abs, cos_weighted_abs)

        # calculate effective pseduo supplier area
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

        # jax.debug.print(
        #     "printing out K from converging func \n\n{} \n\n{} \n\n {}\n",
        #     Ucom,
        #     C_val,
        #     K,
        # )

        # jax.debug.print("printing K shape {}", K.shape)

        return Ucom.reshape(1, 1), K

    # single inlet into two outlets
    def diverging_flow(
        U: jnp.ndarray,
        A: jnp.ndarray,
        theta: jnp.ndarray,
        Q: jnp.ndarray,
    ):
        # jax.debug.print("printing out theta before function starts {}", theta)
        # jax.debug.print("printing out U before function starts {}", U)
        # jax.debug.print("printing out A before function starts {}", A)

        # get supplier and collector segments from the function args
        Q_si = Q[0:1,]
        Q_ci = Q[1:3,]
        U_si = U[0:1,]
        U_ci = U[1:3,]
        # get total flow rate and flow ratio
        Qtot = jnp.sum(Q[0:1,])
        FlowRatio = -Q_ci / Qtot

        # get datum angle
        theta_si = theta[0:1,]
        theta_ci = theta[1:3,]
        # Reorient all branch angles so average collector angle is 0
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
        # calculate pseudosupplier angle
        pseudodirection = jnp.sign(jnp.mean(jnp.sin(theta_si) * Q_si))
        theta = jax.lax.cond(pseudodirection < 0, lambda x: -x, lambda x: x, theta)
        theta_si = theta[0:1,]
        theta_ci = theta[1:3,]

        sin_weighted_abs = jnp.sum(jnp.sin(jnp.abs(theta_si)) * Q_si)
        cos_weighted_abs = jnp.sum(jnp.cos(jnp.abs(theta_si)) * Q_si)
        PseudoSupAngle = jnp.arctan2(sin_weighted_abs, cos_weighted_abs)

        # calculate effective pseduo supplier area
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

        # jax.debug.print(
        #     "printing out K from diverging func \n\n{} \n\n{} \n\n {}\n",
        #     Ucom,
        #     C_val,
        #     K,
        # )

        # jax.debug.print("printing K shape {}", K.shape)

        return Ucom.reshape(1, 1), K

    Q = U * A
    Si = Q >= 0.0
    is_zero_flow = jnp.all(jnp.abs(Q) < 1e-7)
    is_converging = jnp.sum(Si) > 1
    return jax.lax.cond(
        is_zero_flow,
        lambda: zero_flow_case(U, A, theta),
        lambda: jax.lax.cond(
            is_converging,
            lambda: converging_flow(U, A, theta, Q),
            lambda: diverging_flow(U, A, theta, Q),
        ),
    )


@jax.jit
def update_junction_loss_scan_fn(carry, input):
    """
    Updates resistance of daughter vessels associated with a junction. Scanned over stacked arrays of junction parameters
    """
    F, X = carry
    (
        junction_vessel_resistance_index_list,
        junction_nodes_list,
        theta,
        A,
        is_converging,
    ) = input

    flow_rate_index_list = 2 * junction_nodes_list + 1

    Q = X[flow_rate_index_list]
    U = Q / A
    # jax.debug.print("printing out flow-rate {}", U)
    # handle converging or diverging case
    U = jax.lax.cond(
        is_converging, lambda U: U.at[2:].set(-U[2:]), lambda U: U.at[1:].set(-U[1:]), U
    )

    # call the junction loss function for the current junction parameters
    Ucom, K = junction_loss_coefficient(U, A, theta)

    # NOTE: different vals to add based on diverging and converging,note the indexing of K
    val_to_add = jax.lax.cond(
        is_converging,
        # for convering case
        lambda _: jnp.array(
            [
                -0,  # parent vessel
                -0,  # parent vessel
                -0.5 * RHO * Ucom[0, 0] * Ucom[0, 0] * jnp.abs(K[0, 0]),
            ],
            float,
        ).reshape(3, 1),
        # for diverging case
        lambda _: jnp.array(
            [
                -0,  # parent vessel
                -0.5 * RHO * Ucom[0, 0] * Ucom[0, 0] * jnp.abs(K[0, 0]),
                -0.5 * RHO * Ucom[0, 0] * Ucom[0, 0] * jnp.abs(K[1, 0]),
            ],
            float,
        ).reshape(3, 1),
        None,
    )

    # handle division by zero incase flow is stagnant
    val_to_add = jnp.divide(val_to_add, (Q + 1e-10)).reshape(
        3,
    )

    # extract the rows and cols for resistance update in F
    rows, cols = (
        junction_vessel_resistance_index_list[:, 0],
        junction_vessel_resistance_index_list[:, 1],
    )
    # jax.debug.print(
    #     "printing out rows and cols for vessels attached to junction\n {},{},{}\n",
    #     rows,
    #     cols,
    #     is_converging,
    # )
    # finally update F with the junction loss, vectorized operation.
    # jax.debug.print(" \n\nprinting out val_to_add \n\n{}", F[rows, cols])
    F = F.at[rows, cols].set(val_to_add * 1)

    # add dynamic pressure equation, hard coding the values here, if increases stability
    # will implement a better way to make it more general.

    # jax.debug.print("printing from junction scan func")
    return (F, X), None


@jax.jit
def update_all_junctions(
    F: jnp.ndarray,  # global matrix for steady state system
    X: jnp.ndarray,  # current solution for steady state system
    junction_vessel_resistance_index_list_stack: jnp.ndarray,  # all the daughter resistance indices stacked for each junction
    junction_nodes_list_stack: jnp.ndarray,  # junction nodes for each junction stacked along primary axis
    theta_stack: jnp.ndarray,  # angles associated with all junctions stacked along primary axis
    A_stack: jnp.ndarray,  # cross section areas associated with all junctions stacked along primary axis
    is_converging_stack: jnp.ndarray,  # convering boolean flag stacked along primary axis for all junctions in network
):
    # this is the input for the function above, jax.lax.scan will iterate over dim 0 for all components of the
    # scan input.
    scan_inputs = (
        junction_vessel_resistance_index_list_stack,
        junction_nodes_list_stack,
        theta_stack,
        A_stack,
        is_converging_stack,
    )
    (F_after_junction_update, _), _ = jax.lax.scan(
        update_junction_loss_scan_fn, (F, X), scan_inputs
    )

    return F_after_junction_update
