import jax.numpy as jnp
import jax

RHO = 1.06
MU = 0.04


def wrapTopi(angle):
    """Wrap angle to [-pi, pi]"""
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def wrapTo2pi(angle):
    """Wrap angle to [0, 2*pi]"""
    return jnp.mod(angle, 2 * jnp.pi)


def diverging_flow_case_0(
    U: jnp.ndarray,  # velocity at junction
    A: jnp.ndarray,  # areas of branches
    theta: jnp.ndarray,  # angles of all branches wrt datum
    Q: jnp.ndarray,  # flowrate at junction , technically redundant, artifact from previous implementation
):
    """
    Diverging flow case 0: [+ - -]
    Branch 0 is the supplier, branches 1 and 2 are collectors
    """

    # p_loss = Cj*U_j^2 + 1/2*Rho*(U_i^2 - U_j^2)
    loss_coeff = jnp.array((3, 1), float)

    # supplier and collector indices
    si_array = jnp.array([0])
    ci_array = jnp.array([1, 2])

    Q_si = Q[si_array, :]
    Q_ci = Q[ci_array, :]

    U_si = U[si_array, :]
    U_ci = U[ci_array, :]

    Qtot = jnp.sum(Q[si_array, :])

    FlowRatio = -Q_ci / Qtot

    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]

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
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]

    pseudodirection = jnp.sign(jnp.mean(jnp.sin(theta_si) * Q_si))

    theta = jax.lax.cond(pseudodirection < 0, lambda x: -x, lambda x: x, theta)
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]

    sin_weighted_abs = jnp.sum(jnp.sin(jnp.abs(theta_si)) * Q_si)
    cos_weighted_abs = jnp.sum(jnp.cos(jnp.abs(theta_si)) * Q_si)

    PseudoSupAngle = jnp.arctan2(sin_weighted_abs, cos_weighted_abs)

    etransferfactor = (0.8 * (jnp.pi - PseudoSupAngle) * jnp.sign(theta_ci) - 0.2) * (
        1 - FlowRatio
    )

    U_Q_weighted = jnp.sum(U_si * Q_si) / Qtot

    TotPseudoArea = Qtot / ((1 - etransferfactor) * U_Q_weighted)

    A_ci = A[ci_array, :]
    AreaRatio = TotPseudoArea / (A_ci)

    theta = wrapTo2pi(PseudoSupAngle - theta_ci)
    phi = theta

    C_val = (1 - jnp.exp(-FlowRatio / 0.02)) * (
        1 - (1.0 / (AreaRatio * FlowRatio)) * jnp.cos(0.75 * (jnp.pi - phi))
    )
    C_val_to_return = jnp.zeros_like(
        loss_coeff,
    )

    # extremely important piece of code below:
    U0 = U_si[0, 0]
    U1 = U_ci[0, 0]
    U2 = U_ci[1, 0]
    C1_val = C_val[0, 0]
    C2_val = C_val[0, 0]
    C0 = 0.0
    C1 = C1_val * RHO * U1**2 + 0.5 * RHO * (U0**2 - U1**2)
    C2 = C2_val * RHO * U2**2 + 0.5 * RHO * (U0**2 - U2**2)

    C_val_to_return = C_val_to_return.at[0, 0].set(C0)
    C_val_to_return = C_val_to_return.at[1, 0].set(C1)
    C_val_to_return = C_val_to_return.at[2, 0].set(C2)
    jax.debug.print("C_val: {}", C_val_to_return)
    Ucom = U_si
    K = (U_ci**2 / (Ucom**2)) * (2 * C_val + (U_si**2) / (U_ci**2) - 1)

    # jax.debug.print("K,Ucom from diverging flow case 0 {} {}", K, Ucom)

    return Ucom.reshape(1, 1), K


def diverging_flow_case_1(
    U: jnp.ndarray,
    A: jnp.ndarray,
    theta: jnp.ndarray,
    Q: jnp.ndarray,
):
    """
    Diverging flow case 1: [- - +]
    Branch 2 is the supplier, branches 0 and 1 are collectors
    """
    # jax.debug.print("Q  {}", Q)
    # jax.debug.print("from diverging flow case 1")
    # Define index arrays
    si_array = jnp.array([2])
    ci_array = jnp.array([0, 1])

    # Use advanced indexing for all slices
    Q_si = Q[si_array, :]
    Q_ci = Q[ci_array, :]
    U_si = U[si_array, :]
    U_ci = U[ci_array, :]

    # Get total flow rate and flow ratio
    Qtot = jnp.sum(Q[si_array, :])
    FlowRatio = -Q_ci / Qtot

    # Get datum angle
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]

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

    # Use advanced indexing
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]

    # Calculate pseudosupplier angle
    pseudodirection = jnp.sign(jnp.mean(jnp.sin(theta_si) * Q_si))
    theta = jax.lax.cond(pseudodirection < 0, lambda x: -x, lambda x: x, theta)

    # Again use advanced indexing
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]

    sin_weighted_abs = jnp.sum(jnp.sin(jnp.abs(theta_si)) * Q_si)
    cos_weighted_abs = jnp.sum(jnp.cos(jnp.abs(theta_si)) * Q_si)
    PseudoSupAngle = jnp.arctan2(sin_weighted_abs, cos_weighted_abs)

    # Calculate effective pseudo supplier area
    etransferfactor = (0.8 * (jnp.pi - PseudoSupAngle) * jnp.sign(theta_ci) - 0.2) * (
        1 - FlowRatio
    )
    U_Q_weighted = jnp.sum(U_si * Q_si) / Qtot
    TotPseudoArea = Qtot / ((1 - etransferfactor) * U_Q_weighted)

    # Use advanced indexing
    A_ci = A[ci_array, :]

    AreaRatio = TotPseudoArea / (A_ci)
    theta = wrapTo2pi(PseudoSupAngle - theta_ci)
    phi = theta
    C_val = (1 - jnp.exp(-FlowRatio / 0.02)) * (
        1 - (1.0 / (AreaRatio * FlowRatio)) * jnp.cos(0.75 * (jnp.pi - phi))
    )
    Ucom = U_si

    K = (U_ci**2 / (Ucom**2)) * (2 * C_val + (U_si**2) / (U_ci**2) - 1)

    return Ucom.reshape(1, 1), K


def diverging_flow_case_2(
    U: jnp.ndarray,
    A: jnp.ndarray,
    theta: jnp.ndarray,
    Q: jnp.ndarray,
):
    """
    Diverging flow case 2: [- + -]
    Branch 1 is the supplier, branches 0 and 2 are collectors
    """

    # Define index arrays
    si_array = jnp.array([1])
    ci_array = jnp.array([0, 2])

    # Use advanced indexing for all slices
    Q_si = Q[si_array, :]
    Q_ci = Q[ci_array, :]
    U_si = U[si_array, :]
    U_ci = U[ci_array, :]

    # Get total flow rate and flow ratio
    Qtot = jnp.sum(Q[si_array, :])
    FlowRatio = -Q_ci / Qtot

    # Get datum angle
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]

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

    # Use advanced indexing
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]

    # Calculate pseudosupplier angle
    pseudodirection = jnp.sign(jnp.mean(jnp.sin(theta_si) * Q_si))
    theta = jax.lax.cond(pseudodirection < 0, lambda x: -x, lambda x: x, theta)

    # Again use advanced indexing
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]

    sin_weighted_abs = jnp.sum(jnp.sin(jnp.abs(theta_si)) * Q_si)
    cos_weighted_abs = jnp.sum(jnp.cos(jnp.abs(theta_si)) * Q_si)
    PseudoSupAngle = jnp.arctan2(sin_weighted_abs, cos_weighted_abs)

    # Calculate effective pseudo supplier area
    etransferfactor = (0.8 * (jnp.pi - PseudoSupAngle) * jnp.sign(theta_ci) - 0.2) * (
        1 - FlowRatio
    )
    U_Q_weighted = jnp.sum(U_si * Q_si) / Qtot
    TotPseudoArea = Qtot / ((1 - etransferfactor) * U_Q_weighted)

    # Use advanced indexing
    A_ci = A[ci_array, :]

    AreaRatio = TotPseudoArea / (A_ci)
    theta = wrapTo2pi(PseudoSupAngle - theta_ci)
    phi = theta
    C_val = (1 - jnp.exp(-FlowRatio / 0.02)) * (
        1 - (1.0 / (AreaRatio * FlowRatio)) * jnp.cos(0.75 * (jnp.pi - phi))
    )
    Ucom = U_si

    K = (U_ci**2 / (Ucom**2)) * (2 * C_val + (U_si**2) / (U_ci**2) - 1)

    return Ucom.reshape(1, 1), K


def converging_flow_case_0(
    U: jnp.ndarray, A: jnp.ndarray, theta: jnp.ndarray, Q: jnp.ndarray
):
    # jax.debug.print("from converging flow case 0")
    si_array = jnp.array([1, 2])
    ci_array = jnp.array([0])
    Q_si = Q[si_array, :]
    Q_ci = Q[ci_array, :]
    U_si = U[si_array, :]
    U_ci = U[ci_array, :]
    Qtot = jnp.sum(Q[si_array, :])
    FlowRatio = -Q_ci / Qtot
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]
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
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]
    pseudodirection = jnp.sign(jnp.mean(jnp.sin(theta_si) * Q_si))
    theta = jax.lax.cond(pseudodirection < 0, lambda x: -x, lambda x: x, theta)
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]
    sin_weighted_abs = jnp.sum(jnp.sin(jnp.abs(theta_si)) * Q_si)
    cos_weighted_abs = jnp.sum(jnp.cos(jnp.abs(theta_si)) * Q_si)
    PseudoSupAngle = jnp.arctan2(sin_weighted_abs, cos_weighted_abs)
    etransferfactor = (0.8 * (jnp.pi - PseudoSupAngle) * jnp.sign(theta_ci) - 0.2) * (
        1 - FlowRatio
    )
    U_Q_weighted = jnp.sum(U_si * Q_si) / Qtot
    TotPseudoArea = Qtot / ((1 - etransferfactor) * U_Q_weighted)
    A_ci = A[ci_array, :]
    AreaRatio = TotPseudoArea / (A_ci)
    theta = wrapTo2pi(PseudoSupAngle - theta_ci)
    phi = theta
    C_val = (1 - jnp.exp(-FlowRatio / 0.02)) * (
        1 - (1.0 / (AreaRatio * FlowRatio)) * jnp.cos(0.75 * (jnp.pi - phi))
    )
    jax.debug.print("C_val: {}", C_val.shape)
    Ucom = U_ci
    K = (U_si**2 / (Ucom**2)) * (2 * C_val + (U_ci**2) / (U_si**2) - 1)
    # jax.debug.print("K,Ucom from converging flow case 0 {} {}", K, Ucom)
    return Ucom.reshape(1, 1), K


def converging_flow_case_1(
    U: jnp.ndarray, A: jnp.ndarray, theta: jnp.ndarray, Q: jnp.ndarray
):
    # jax.debug.print("from converging flow case 1")
    si_array = jnp.array([0, 2])
    ci_array = jnp.array([1])
    Q_si = Q[si_array, :]
    Q_ci = Q[ci_array, :]
    U_si = U[si_array, :]
    U_ci = U[ci_array, :]
    Qtot = jnp.sum(Q[si_array, :])
    FlowRatio = -Q_ci / Qtot
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]
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
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]
    pseudodirection = jnp.sign(jnp.mean(jnp.sin(theta_si) * Q_si))
    theta = jax.lax.cond(pseudodirection < 0, lambda x: -x, lambda x: x, theta)
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]
    sin_weighted_abs = jnp.sum(jnp.sin(jnp.abs(theta_si)) * Q_si)
    cos_weighted_abs = jnp.sum(jnp.cos(jnp.abs(theta_si)) * Q_si)
    PseudoSupAngle = jnp.arctan2(sin_weighted_abs, cos_weighted_abs)
    etransferfactor = (0.8 * (jnp.pi - PseudoSupAngle) * jnp.sign(theta_ci) - 0.2) * (
        1 - FlowRatio
    )
    U_Q_weighted = jnp.sum(U_si * Q_si) / Qtot
    TotPseudoArea = Qtot / ((1 - etransferfactor) * U_Q_weighted)
    A_ci = A[ci_array, :]
    AreaRatio = TotPseudoArea / (A_ci)
    theta = wrapTo2pi(PseudoSupAngle - theta_ci)
    phi = theta
    C_val = (1 - jnp.exp(-FlowRatio / 0.02)) * (
        1 - (1.0 / (AreaRatio * FlowRatio)) * jnp.cos(0.75 * (jnp.pi - phi))
    )
    Ucom = U_ci
    K = (U_si**2 / (Ucom**2)) * (2 * C_val + (U_ci**2) / (U_si**2) - 1)
    return Ucom.reshape(1, 1), K


def converging_flow_case_2(
    U: jnp.ndarray, A: jnp.ndarray, theta: jnp.ndarray, Q: jnp.ndarray
):
    # jax.debug.print("U  {}", Q)
    # jax.debug.print("from converging flow case 2")
    si_array = jnp.array([0, 1])
    ci_array = jnp.array([2])
    Q_si = Q[si_array, :]
    Q_ci = Q[ci_array, :]
    U_si = U[si_array, :]
    U_ci = U[ci_array, :]
    Qtot = jnp.sum(Q[si_array, :])
    FlowRatio = -Q_ci / Qtot
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]
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
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]
    pseudodirection = jnp.sign(jnp.mean(jnp.sin(theta_si) * Q_si))
    theta = jax.lax.cond(pseudodirection < 0, lambda x: -x, lambda x: x, theta)
    theta_si = theta[si_array, :]
    theta_ci = theta[ci_array, :]
    sin_weighted_abs = jnp.sum(jnp.sin(jnp.abs(theta_si)) * Q_si)
    cos_weighted_abs = jnp.sum(jnp.cos(jnp.abs(theta_si)) * Q_si)
    PseudoSupAngle = jnp.arctan2(sin_weighted_abs, cos_weighted_abs)
    etransferfactor = (0.8 * (jnp.pi - PseudoSupAngle) * jnp.sign(theta_ci) - 0.2) * (
        1 - FlowRatio
    )
    U_Q_weighted = jnp.sum(U_si * Q_si) / Qtot
    TotPseudoArea = Qtot / ((1 - etransferfactor) * U_Q_weighted)
    A_ci = A[ci_array, :]
    AreaRatio = TotPseudoArea / (A_ci)
    theta = wrapTo2pi(PseudoSupAngle - theta_ci)
    phi = theta
    C_val = (1 - jnp.exp(-FlowRatio / 0.02)) * (
        1 - (1.0 / (AreaRatio * FlowRatio)) * jnp.cos(0.75 * (jnp.pi - phi))
    )
    Ucom = U_ci
    K = (U_si**2 / (Ucom**2)) * (2 * C_val + (U_ci**2) / (U_si**2) - 1)
    return Ucom.reshape(1, 1), K


def zero_flow_case(U: jnp.ndarray, A: jnp.ndarray, theta: jnp.ndarray, Q: jnp.ndarray):
    # jax.debug.print("from zero flow case")
    dummy_Ucom = jnp.zeros((1, 1))
    dummy_K = jnp.zeros((2, 1))
    return dummy_Ucom, dummy_K


@jax.jit
def junction_loss_coefficient(U: jnp.ndarray, A: jnp.ndarray, theta: jnp.ndarray):
    Q = U * A
    is_zero_flow = jnp.any(jnp.abs(Q) < 1e-7)

    # diverging flows
    is_diverging_0 = jnp.all((Q[0] > 0) & (Q[1] < 0) & (Q[2] < 0))  # [+ - -]
    is_diverging_1 = jnp.all((Q[0] < 0) & (Q[1] < 0) & (Q[2] > 0))  # [- - +]
    is_diverging_2 = jnp.all((Q[0] < 0) & (Q[1] > 0) & (Q[2] < 0))  # [- + -]

    # converging flow
    is_converging_0 = jnp.all((Q[0] < 0) & (Q[1] > 0) & (Q[2] > 0))  # [- + +]
    is_converging_1 = jnp.all((Q[0] > 0) & (Q[1] > 0) & (Q[2] < 0))  # [+ + -]
    is_converging_2 = jnp.all((Q[0] > 0) & (Q[1] < 0) & (Q[2] > 0))  # [+ - +]

    case_index = is_zero_flow * 6 + (1 - is_zero_flow) * (
        is_diverging_0 * 0
        + is_diverging_1 * 1
        + is_diverging_2 * 2
        + is_converging_0 * 3
        + is_converging_1 * 4
        + is_converging_2 * 5
        + (
            1
            - (
                is_diverging_0
                | is_diverging_1
                | is_diverging_2
                | is_converging_0
                | is_converging_1
                | is_converging_2
            )
        )
        * 6
    )

    return jax.lax.switch(
        case_index,
        [
            lambda args: diverging_flow_case_0(*args),  # Case 0
            lambda args: diverging_flow_case_1(*args),  # Case 1
            lambda args: diverging_flow_case_2(*args),  # Case 2
            lambda args: converging_flow_case_0(*args),  # Case 3
            lambda args: converging_flow_case_1(*args),  # Case 4
            lambda args: converging_flow_case_2(*args),  # Case 5
            lambda args: zero_flow_case(*args),  # Case 6 - Zero flow
        ],
        (U, A, theta, Q),
    )
