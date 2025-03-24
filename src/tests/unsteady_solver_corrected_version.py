import numpy
import numpy as np
from numpy import *
import scipy
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve
from scipy import interpolate


# temp import to suppress warnings the interpreter vomits to console
import warnings

warnings.filterwarnings("ignore")


def fft_data(T, dt, A):
    N = 50
    t = A[:, 0]
    f = A[:, 1]

    t_even = np.linspace(0, T, N + 1)
    t_even = t_even[0:-1]
    f_even = scipy.interpolate.pchip_interpolate(t, f, t_even, der=0, axis=0)

    X = np.fft.fft(f_even)
    X = np.array([X]).T

    t_fft = np.linspace(0, T, 300)
    f_fft = np.zeros((300, 1))
    dfdt_fft = np.zeros((300, 1))

    if np.mod(N, 2) == 0:
        # added floor division operator to get int in the linspace arg
        n = np.concatenate(
            (
                np.array([np.linspace(0, N // 2 - 1, N // 2)]),
                np.array([np.linspace(-N // 2, -1, N // 2)]),
            ),
            axis=1,
        ).T
    else:
        n = np.concatenate(
            (
                np.array([np.linspace(0, int(N / 2), int(N / 2) + 1)]),
                np.array([np.linspace(-int(N / 2), -1, int(N / 2))]),
            ),
            axis=1,
        ).T

    for k in range(0, N):
        f_fft = (
            f_fft
            + np.array(
                [X[k, 0] * np.exp(2 * np.pi * 1j * n[k, 0] * t_fft * N / T / N)]
            ).T
        )
    f_fft = np.real(f_fft) / N

    N = X.shape[0]
    if np.mod(N, 2) == 0:
        # added floor-division operator to get int in the linspace arg
        n = np.concatenate(
            (
                np.array([np.linspace(0, N / 2 - 1, N // 2)]),
                np.array([np.linspace(-N / 2, -1, N // 2)]),
            ),
            axis=1,
        ).T
    else:
        n = np.concatenate(
            (
                np.array([np.linspace(0, int(N / 2), int(N / 2) + 1)]),
                np.array([np.linspace(-int(N / 2), -1, int(N / 2))]),
            ),
            axis=1,
        ).T

    c1 = 0
    tt = 0
    step = dt
    QQ = np.zeros((int(T / step) + 1, 1))
    TT = np.zeros((int(T / step) + 1, 1))
    for i in range(0, int(T / step) + 1):
        QQ[c1, 0] = np.real(np.sum(X * np.exp(2.0 * np.pi * 1j * n * tt / T))) / N
        TT[c1, 0] = tt
        c1 = c1 + 1
        tt = tt + dt
    return QQ


def junction_loss_coefficient(U, A, theta):
    xwrap = np.remainder(theta, 2 * np.pi)
    mask = np.abs(xwrap) > np.pi
    xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
    theta = xwrap
    print(f"U: {U}, A: {A}, theta: {theta}")
    if U[0, 0] < 1e-7:
        Ucom = np.array([[0]])
        K = np.array([[0], [0]])
    else:
        Q = U * A
        Ci = Q < 0.0
        Si = Q >= 0.0
        Qtot = np.sum(Q[Si])
        FlowRatio = -Q[Ci] / Qtot
        PseudoColAngle = np.mean(theta[Ci])
        PseudoSupAngle = math.atan2(
            np.sum(sin(theta[Si]) * Q[Si]), np.sum((cos(theta[Si])) * Q[Si])
        )
        if np.abs(PseudoSupAngle - PseudoColAngle) < 0.5 * np.pi:
            PseudoColAngle = PseudoColAngle + np.pi
        xwrap = np.remainder(theta - PseudoColAngle, 2 * np.pi)
        mask = np.abs(xwrap) > np.pi
        xwrap[mask] -= 2 * np.pi * np.sign(xwrap[mask])
        theta = xwrap
        pseudodirection = np.sign(np.mean(sin(theta[Si]) * Q[Si]))
        if pseudodirection < 0:
            theta = -theta

        PseudoSupAngle = math.atan2(
            np.sum(sin(np.abs(theta[Si])) * Q[Si]),
            np.sum(cos(np.abs(theta[Si])) * Q[Si]),
        )
        etransferfactor = (
            0.8 * (np.pi - PseudoSupAngle) * np.sign(theta[Ci]) - 0.2
        ) * (1 - FlowRatio)
        TotPseudoArea = Qtot / ((1 - etransferfactor) * np.sum(U[Si] * Q[Si]) / Qtot)
        AreaRatio = TotPseudoArea / A[Ci]
        xwrap = np.remainder(PseudoSupAngle - theta[Ci], 4 * np.pi)
        mask = np.abs(xwrap) > 2 * np.pi
        xwrap[mask] -= 4 * np.pi * np.sign(xwrap[mask])
        phi = xwrap
        C = np.zeros((U.shape[0], 1))
        C[Ci] = (1 - np.exp(-FlowRatio / 0.02)) * (
            1 - (1.0 / (AreaRatio * FlowRatio)) * cos(0.75 * (np.pi - phi))
        )

        if np.sum(Ci) == 1:
            Ucom = U[Ci]
        else:
            Ucom = U[Si]
        K = (U[Ci] ** 2 / Ucom**2) * (2 * C[Ci] + U[Si] ** 2 / U[Ci] ** 2 - 1)
    # print(f"Ucom: {Ucom}, K: {K}")
    return [np.array([Ucom]), np.array([K]).T.reshape(2, 1)]
    # return [Ucom.reshape((1, 1)), K.reshape(2, 1)]


def viscous_kinetic_loss_coefficient(x_f, r_f, mu, rho):
    Kv = 1.0
    Kt = 1.52
    dr = np.diff(r_f, axis=0)
    idx = numpy.array([], dtype=int)
    idn = numpy.array([], dtype=int)
    rmax = numpy.array([], dtype=float)
    rmin = numpy.array([], dtype=float)
    for i in range(0, dr.shape[0] - 1):
        if dr[i, 0] > 0:
            if dr[i + 1, 0] < 0:
                idx = np.append(idx, int(i + 1))
                rmax = np.append(rmax, r_f[i + 1, 0])
            elif dr[i + 1, 0] == 0:
                for j in range(i + 2, dr.shape[0] - 1):
                    if dr[j, 0] > 0:
                        break
                    elif dr[j, 0] < 0:
                        idx = np.append(idx, int(i + 1))
                        rmax = np.append(rmax, r_f[i + 1, 0])
                        break
        if dr[i, 0] < 0:
            if dr[i + 1, 0] > 0:
                idn = np.append(idn, int(i + 1))
                rmin = np.append(rmin, r_f[i + 1, 0])
            elif dr[i + 1, 0] == 0:
                for j in range(i + 2, dr.shape[0] - 1):
                    if dr[j, 0] < 0:
                        break
                    elif dr[j, 0] > 0:
                        idn = np.append(idn, int(i + 1))
                        rmin = np.append(rmin, r_f[i + 1, 0])
                        break
    idx = np.array([idx]).T
    idn = np.array([idn]).T
    rmax = np.array([rmax]).T
    rmin = np.array([rmin]).T
    s1 = idn.shape[0]
    s2 = idx.shape[0]
    C2 = 0
    C1 = 0
    Resi = 0
    if s1 == 0:
        Resi_m = np.zeros((1, 1))
        C2_m = np.zeros((1, 1))
    else:
        Resi_m = np.zeros((s1 + 1, 1))
        C2_m = np.zeros((s1, 1))
    c14 = 0
    if s1 == 0:
        C2 = 0
        C1 = 0
        Resi = (8 * mu / pi) * np.trapz(r_f ** (-4), x_f, axis=0)
        C2_m[0] = C2
        Resi_m[0] = Resi
        C2_m1 = C2_m
        IDX = np.array([])
        c14 = c14 + 1
    elif s1 == s2:
        IDX = np.zeros((2 * s1, 1))
        if np.max(idn) > np.max(idx):
            IDX[0::2] = idx
            IDX[1::2] = idn
            mnxe = 1
            mnxi = 0
            for i in range(0, s1):
                if i != s1 - 1:
                    As = pi * rmin[i, 0] * rmin[i, 0]
                    An = np.max(
                        [pi * rmax[i] * rmax[i], pi * rmax[i + 1] * rmax[i + 1]]
                    )
                    C2 = C2 + rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
                    C2_m[c14, 0] = rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
                    c14 = c14 + 1
                else:
                    As = pi * rmin[i] * rmin[i]
                    # error here
                    # slicing operations on r_f changed, added item() func call to change (1,) array to float for np.max to work
                    An = np.max(
                        [
                            pi * rmax[i].item() * rmax[i].item(),
                            pi
                            * np.max(r_f[int(idn[i]) :, 0])
                            * np.max(r_f[int(idn[i]) :, 0]),
                        ]
                    )
                    C2 = C2 + rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
                    C2_m[c14, 0] = rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
                    c14 = c14 + 1

            C1 = (
                Kv
                * (8 * mu / pi)
                * np.trapz(r_f[int(idx[0]) :] ** (-4), x_f[int(idx[0]) :], axis=0)
            )
            Resi = (8 * mu / pi) * np.trapz(
                r_f[0 : int(idx[0]) + 1] ** (-4), x_f[0 : int(idx[0]) + 1], axis=0
            )

        elif np.max(idn) < np.max(idx):
            IDX[0::2] = idn
            IDX[1::2] = idx
            mnxe = 0
            mnxi = 1
            for i in range(0, s1):
                if i > 0:
                    As = pi * rmin[i] * rmin[i]
                    An = np.max(
                        [pi * rmax[i] * rmax[i], pi * rmax[i - 1] * rmax[i - 1]]
                    )
                    C2 = C2 + rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
                    C2_m[c14, 0] = rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
                    c14 = c14 + 1
                else:
                    As = pi * rmin[i] * rmin[i]
                    An = np.max(
                        [
                            pi * rmax[i] * rmax[i],
                            pi
                            * np.max(r_f[0 : int(idn[i]) + 1])
                            * np.max(r_f[0 : int(idn[i]) + 1]),
                        ]
                    )
                    C2 = C2 + rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
                    C2_m[c14, 0] = rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
                    c14 = c14 + 1

            C1 = (
                Kv
                * (8 * mu / pi)
                * np.trapz(
                    r_f[0 : int(idx[-1]) + 1] ** (-4), x_f[0 : int(idx[-1]) + 1], axis=0
                )
            )
            Resi = (8 * mu / pi) * np.trapz(
                r_f[int(idx[-1]) :] ** (-4), x_f[int(idx[-1]) :], axis=0
            )

    elif s1 > s2:
        IDX = np.zeros((s1 + s2, 1))
        if s1 > 1:
            IDX[0::2] = idn
            IDX[1::2] = idx
            mnxe = 1
            mnxi = 1
            for i in range(0, s1):
                if 0 < i and i < s1 - 1:
                    As = pi * rmin[i] * rmin[i]
                    An = np.max(
                        [pi * rmax[i] * rmax[i], pi * rmax[i - 1] * rmax[i - 1]]
                    )
                    C2 = C2 + rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
                    C2_m[c14, 0] = rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
                    c14 = c14 + 1
                elif i == s1 - 1:
                    As = pi * rmin[i] * rmin[i]
                    An = np.max(
                        [
                            pi * rmax[i - 1].item() * rmax[i - 1].item(),
                            pi
                            * np.max(r_f[int(idn[i]) :])
                            * np.max(r_f[int(idn[i]) :]),
                        ]
                    )
                    C2 = C2 + rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
                    C2_m[c14, 0] = rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
                    c14 = c14 + 1
                elif i == 0:
                    As = pi * rmin[i] * rmin[i]
                    # added item() func call to convert (1,) shaped array to a float
                    An = np.max(
                        [
                            pi * rmax[i].item() * rmax[i].item(),
                            pi
                            * np.max(r_f[0 : int(idn[i]) + 1])
                            * np.max(r_f[0 : int(idn[i]) + 1]),
                        ]
                    )
                    C2 = C2 + rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
                    C2_m[c14, 0] = rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
                    c14 = c14 + 1

        else:
            IDX[0] = idn
            mnxe = 1
            mnxi = 1
            As = pi * rmin[0] * rmin[0]
            An = pi * np.max(r_f) * np.max(r_f)
            C2 = C2 + rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
            C2_m[c14, 0] = rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
            c14 = c14 + 1
        C1 = Kv * (8 * mu / pi) * np.trapz(r_f ** (-4), x_f, axis=0)
        Resi = 0

    elif s1 < s2:
        IDX = np.zeros((s1 + s2, 1))
        IDX[0::2] = idx
        IDX[1::2] = idn
        mnxe = 0
        mnxi = 0
        for i in range(0, s1):
            As = pi * rmin[i] * rmin[i]
            An = np.max([pi * rmax[i + 1] * rmax[i + 1], pi * rmax[i] * rmax[i]])
            C2 = C2 + rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
            C2_m[c14, 0] = rho * Kt * (((An / As) - 1) ** 2) / 2 / An / An
            c14 = c14 + 1
        C1 = (
            Kv
            * (8 * mu / pi)
            * np.trapz(
                r_f[int(idx[0]) : int(idx[-1]) + 1] ** (-4),
                x_f[int(idx[0]) : int(idx[-1]) + 1],
                axis=0,
            )
        )
        Resi = (8 * mu / pi) * np.trapz(
            r_f[0 : int(idx[0]) + 1] ** (-4), x_f[0 : int(idx[0]) + 1], axis=0
        ) + (8 * mu / pi) * np.trapz(
            r_f[int(idx[-1]) :] ** (-4), x_f[int(idx[-1]) :], axis=0
        )

    if s1 == 0:
        Resi_m[0, 0] = Resi

    else:
        Resi_m = np.zeros(((s1 - 1) * 2 + 2, 1))
        C2_m1 = np.zeros(((s1 - 1) * 2 + 2, 1))

        if mnxi == 1:
            Resi_m[0, 0] = (8 * mu / pi) * np.trapz(
                r_f[0 : int(IDX[0]) + 1] ** (-4), x_f[0 : int(IDX[0]) + 1], axis=0
            )

        else:
            IDX = IDX[1:]
            Resi_m[0, 0] = (8 * mu / pi) * np.trapz(
                r_f[0 : int(IDX[0]) + 1] ** (-4), x_f[0 : int(IDX[0]) + 1], axis=0
            )

        if mnxe == 1:
            Resi_m[-1, 0] = (8 * mu / pi) * np.trapz(
                r_f[int(IDX[-1]) :] ** (-4), x_f[int(IDX[-1]) :], axis=0
            )
        else:
            Resi_m[-1, 0] = (8 * mu / pi) * np.trapz(
                r_f[int(IDX[-2]) :] ** (-4), x_f[int(IDX[-2]) :], axis=0
            )

        for i in range(1, (s1 - 1) * 2 + 1):
            Resi_m[i, 0] = (8 * mu / pi) * np.trapz(
                r_f[int(IDX[i - 1]) : int(IDX[i]) + 1] ** (-4),
                x_f[int(IDX[i - 1]) : int(IDX[i]) + 1],
                axis=0,
            )

        if mnxe < 0.5:
            IDX = IDX[0:-1]
        C2_m1[1::2] = C2_m

    return [IDX, Resi_m, C2_m1, np.array([Resi]), np.array([C1]), np.array([C2])]


# load in  stuff
bif = numpy.loadtxt("./all_files_old/bifurcation.dat", dtype=int)
trif = numpy.loadtxt("./all_files_old/trifurcation.dat", dtype=int)

acl = numpy.loadtxt("./all_files_old/area_curv_length_idealized.dat", dtype=float)
leng = numpy.loadtxt("./all_files_old/length.dat", dtype=float)
resi = numpy.loadtxt("./all_files_old/resistance.dat", dtype=float)
ang = numpy.loadtxt("./all_files_old/angles.dat", dtype=float)
numread = numpy.loadtxt("./all_files_old/numread.dat", dtype=int)


T = 1.1
np1 = 20
dt = 0.001

inlet_ind = np.array([[0.0]])
outlet_ind = numpy.loadtxt("./all_files_old/outlet_ind.dat", dtype=float)
NO = outlet_ind.shape[0]
NT = 0
NQ = 0
NJ = 1
NV = 3

NR = outlet_ind.shape[0]
mu = 0.04
rho = 1.06
Pin = 100 / 0.00075 * 1


aorta_flow = numpy.loadtxt("./all_files_old/aorta-flow_quasi_steady.dat", dtype=float)
Q_inll = fft_data(T, dt, aorta_flow)
q0 = 1e-1
q1 = 1e-1
q2 = 1e-1
p01 = Pin
p1o = Pin
p2o = Pin
pin = Pin
bif = np.array([bif])
ang = np.array([ang])


r = np.zeros((NV, 1))
R = np.zeros((NV, 1))
R_old = np.zeros((NV, 1))
CS = np.cumsum(numread)
# adding a bunch of print statements for debug stuff
print(CS)
# print(CS.shape)
# print(np.array([0]).shape)
# print()
# print()
# changing axis parameter to None
CS = np.concatenate((np.array([0]), CS), axis=None)
# print(CS)
# print()
CS = np.array([CS])
CS = CS.T

area = np.array([acl[CS[0, 0] : CS[1, 0], 0]])
area = area.T
l = np.array([acl[CS[0, 0] : CS[1, 0], 2]])
l = l.T
l10 = np.concatenate((np.array([[0]]), l, np.array([[leng[0]]])), axis=0)
l1, u1 = np.unique(l10, return_index=True)
l1 = np.array([l1])
l1 = l1.T

# adding debug print statements
# print(l1.shape)
# print()

area1 = np.concatenate(
    (np.array([[area[0, 0]]]), area, np.array([[area[-1, 0]]])), axis=0
)
r1 = np.sqrt(area1[u1] / pi)
# adding debug print statements
# print(r1.shape)
[idn, Resi_m, C2_m, Resi, C1, C2] = viscous_kinetic_loss_coefficient(l1, r1, mu, rho)
idn = idn.astype(int)
idn_0 = idn
C2_1 = C2
C2_m0 = C2_m
Resi_m0 = Resi_m
R[0, 0] = Resi + C1
l1_m0 = l1[idn.tolist(), 0]
if idn.shape[0] == 0:
    l_m0 = np.concatenate(
        (np.array([[l1[0, 0]]]), np.array([[l1[-1, 0]]]), np.zeros((50 - 2, 1))), axis=0
    )
else:
    l_m0 = np.concatenate(
        (
            np.array([[l1[0, 0]]]),
            l1_m0,
            np.array([[l1[-1, 0]]]),
            np.zeros((50 - 2 - l1_m0.shape[0], 1)),
        ),
        axis=0,
    )
r[0, 0] = ((8 * mu / pi) * l1[-1, 0] / (Resi + C1)) ** 0.25
R_old[1, 0] = (8 * mu / pi) * leng[0] / (r[0, 0] ** 4)


area = np.array([acl[CS[1, 0] : CS[2, 0], 0]])
area = area.T
l = np.array([acl[CS[1, 0] : CS[2, 0], 2]])
l = l.T
l10 = np.concatenate((np.array([[0]]), l, np.array([[leng[1]]])), axis=0)
l1, u1 = np.unique(l10, return_index=True)
l1 = np.array([l1])
l1 = l1.T
area1 = np.concatenate(
    (np.array([[area[0, 0]]]), area, np.array([[area[-1, 0]]])), axis=0
)
r1 = np.sqrt(area1[u1] / pi)
[idn, Resi_m, C2_m, Resi, C1, C2] = viscous_kinetic_loss_coefficient(l1, r1, mu, rho)
idn = idn.astype(int)
idn_1 = idn
C2_0 = C2
C2_m1 = C2_m
Resi_m1 = Resi_m
R[1, 0] = Resi + C1
l1_m1 = l1[idn.tolist(), 0]
if idn.shape[0] == 0:
    l_m1 = np.concatenate(
        (np.array([[l1[0, 0]]]), np.array([[l1[-1, 0]]]), np.zeros((50 - 2, 1))), axis=0
    )
else:
    l_m1 = np.concatenate(
        (
            np.array([[l1[0, 0]]]),
            l1_m1,
            np.array([[l1[-1, 0]]]),
            np.zeros((50 - 2 - l1_m1.shape[0], 1)),
        ),
        axis=0,
    )
r[1, 0] = ((8 * mu / pi) * l1[-1, 0] / (Resi + C1)) ** 0.25
R_old[1, 0] = (8 * mu / pi) * leng[1] / (r[1, 0] ** 4)


area = np.array([acl[CS[2, 0] : CS[3, 0], 0]])
area = area.T
l = np.array([acl[CS[2, 0] : CS[3, 0], 2]])
l = l.T
l10 = np.concatenate((np.array([[0]]), l, np.array([[leng[2]]])), axis=0)
l1, u1 = np.unique(l10, return_index=True)
l1 = np.array([l1])
l1 = l1.T
area1 = np.concatenate(
    (np.array([[area[0, 0]]]), area, np.array([[area[-1, 0]]])), axis=0
)
r1 = np.sqrt(area1[u1] / pi)
[idn, Resi_m, C2_m, Resi, C1, C2] = viscous_kinetic_loss_coefficient(l1, r1, mu, rho)
idn = idn.astype(int)
idn_2 = idn
C2_2 = C2
C2_m2 = C2_m
Resi_m2 = Resi_m
R[2, 0] = Resi + C1
l1_m2 = l1[idn.tolist(), 0]
if idn.shape[0] == 0:
    l_m2 = np.concatenate(
        (np.array([[l1[0, 0]]]), np.array([[l1[-1, 0]]]), np.zeros((50 - 2, 1))), axis=0
    )
else:
    l_m2 = np.concatenate(
        (
            np.array([[l1[0, 0]]]),
            l1_m2,
            np.array([[l1[-1, 0]]]),
            np.zeros((50 - 2 - l1_m2.shape[0], 1)),
        ),
        axis=0,
    )
r[2, 0] = ((8 * mu / pi) * l1[-1, 0] / (Resi + C1)) ** 0.25
R_old[2, 0] = (8 * mu / pi) * leng[2] / (r[2, 0] ** 4)


R0 = np.array([resi]).T

s = 1
X0 = np.array([[q0], [q1], [q2], [p01], [p1o], [p2o], [pin]])

X_1 = X0
X_2 = X0
X0_t = X0
X = np.zeros((NV + NJ + NT + NQ + NO + 1, 1))
Mo = np.zeros((NV, 1))
bif_dis = np.zeros((NJ, 3))
M1 = np.zeros((NV, NV))
M2 = np.zeros((NV, NJ + NT + NQ + NO + 1))
M3 = np.zeros((NJ + NT + NQ + NO + 1, NV + NJ + NT + NQ + NO + 1))
b = np.zeros((NV + NJ + NT + NQ + NO + 1, 1))

P1 = np.zeros((100 * np1 + 1, 1))
P2 = np.zeros((100 * np1 + 1, 1))
Q1 = np.zeros((100 * np1 + 1, 1))
Q2 = np.zeros((100 * np1 + 1, 1))
PJ01 = np.zeros((100 * np1 + 1, 1))
P0 = np.zeros((100 * np1 + 1, 1))
Q0 = np.zeros((100 * np1 + 1, 1))
tt = np.zeros((100 * np1 + 1, 1))
c4 = 0

# added a int func call to convert float to int
Qin = np.zeros((int(np1 * T / dt + 1), 1))
Qin[0 : int(T / dt + 1), 0] = Q_inll[:, 0]
for i in range(0, np1):
    Qin[int(i * T / dt + 1) : int(i * T / dt + 1 + T / dt), 0] = Q_inll[1:, 0]
Qin = Qin + 1e-3
print(R)
print(C2_0)
print(C2_1)
print(C2_2)
# breakpoint()
# plot inlet flow rate waveform
# plt.plot(Qin)
# plt.show()
# breakpoint()

# adding print debug statements to see the size of bif
# print(bif.shape)
# print(NJ)
# print(ang.shape)
# solver start

Pin = np.zeros_like(Qin)
Pout1 = np.zeros_like(Qin)
Pout2 = np.zeros_like(Qin)
Qout1 = np.zeros_like(Qin)
Qout2 = np.zeros_like(Qin)
for c in range(0, int(np1 * T / dt) + 1):
    s = 1
    k = 0
    while s:
        k = k + 1

        for i in range(0, NV):
            area = np.array([acl[CS[i, 0] : CS[i + 1, 0], 0]])
            area = area.T
            curv = np.array([acl[CS[i, 0] : CS[i + 1, 0], 1]])
            curv = curv.T
            if np.mean(curv) < 0 or np.mean(curv) > 50:
                Mo[i, 0] = 1
            else:
                ri = np.sqrt(area / pi)
                De = ((2 * rho * X0[i, 0] / pi / mu) / ri) * (np.sqrt(ri / curv))
                if np.mean(De[1:-1, 0]) > 10:
                    mo = (
                        0.1008
                        * (np.sqrt(De))
                        * (np.sqrt(1 + 1.729 / De) - 1.315 / (np.sqrt(De))) ** (-3)
                    )
                    Mo[i, 0] = np.mean(mo[1:-1, 0])
                else:
                    Mo[i, 0] = 1
        for i in range(0, NJ):
            i1 = bif[i, 0]
            i2 = bif[i, 1]
            i3 = bif[i, 2]
            [Ucom, K] = junction_loss_coefficient(
                np.array(
                    [
                        [X0[i1, 0] / np.pi / r[i1, 0] / r[i1, 0]],
                        [-X0[i2, 0] / np.pi / r[i2, 0] / r[i2, 0]],
                        [-X0[i3, 0] / np.pi / r[i3, 0] / r[i3, 0]],
                    ]
                ),
                np.array(
                    [
                        [np.pi * r[i1, 0] * r[i1, 0]],
                        [np.pi * r[i2, 0] * r[i2, 0]],
                        [np.pi * r[i3, 0] * r[i3, 0]],
                    ]
                ),
                np.array(
                    [[np.pi], [math.radians(ang[i, 0])], [-math.radians(ang[i, 1])]]
                ),
            )
            print(f"Ucom: {Ucom}, K: {K}")
            bif_dis[i, 0] = Ucom[0, 0]
            bif_dis[i, 1] = np.abs(K[0, 0])
            bif_dis[i, 2] = np.abs(K[1, 0])
        # print("end")

        for i in range(0, NV):
            # print(R[i, 0])
            M1[i, i] = -Mo[i, 0] * R[i, 0]

        for i in range(0, NJ):
            i2 = bif[i, 1]
            i3 = bif[i, 2]
            M1[i2, i2] = (
                M1[i2, i2]
                - 0.5
                * rho
                * bif_dis[i, 0]
                * bif_dis[i, 0]
                * bif_dis[i, 1]
                / X0[i2, 0]
                * 1
            )
            # print(
            #     -0.5 * rho * bif_dis[i, 0] * bif_dis[i, 0] * bif_dis[i, 1] / X0[i2, 0]
            # )
            # print(
            #     -0.5 * rho * bif_dis[i, 0] * bif_dis[i, 0] * bif_dis[i, 2] / X0[i3, 0]
            # )
            M1[i3, i3] = (
                M1[i3, i3]
                - 0.5
                * rho
                * bif_dis[i, 0]
                * bif_dis[i, 0]
                * bif_dis[i, 2]
                / X0[i3, 0]
                * 1
            )

        b[0, 0] = -1 / dt * X_1[0, 0]
        b[1, 0] = -1 / dt * X_1[1, 0]
        b[2, 0] = -1 / dt * X_1[2, 0]

        M1[0, 0] = M1[0, 0] - 2 * C2_0 * X_1[0, 0] - 1 / dt
        M1[1, 1] = M1[1, 1] - 2 * C2_1 * X_1[1, 0] - 1 / dt
        M1[2, 2] = M1[2, 2] - 2 * C2_2 * X_1[2, 0] - 1 / dt
        # print(C2_0)
        b[0, 0] = b[0, 0] - C2_0 * X_1[0, 0] * X_1[0, 0]
        b[1, 0] = b[1, 0] - C2_1 * X_1[1, 0] * X_1[1, 0]
        b[2, 0] = b[2, 0] - C2_2 * X_1[2, 0] * X_1[2, 0]
        for i in range(0, NJ):
            i1 = bif[i, 0]
            i2 = bif[i, 1]
            i3 = bif[i, 2]
            M2[i1, i + NT + NQ] = -1
            M2[i2, i + NT + NQ] = 1
            M2[i3, i + NT + NQ] = 1

        for i in range(0, NJ):
            i1 = bif[i, 0]
            i2 = bif[i, 1]
            i3 = bif[i, 2]
            M3[i + NT + NQ, i1] = 1
            M3[i + NT + NQ, i2] = -1
            M3[i + NT + NQ, i3] = -1

        for i in range(0, NO):
            ## adding cast to int for func expecting int instead of float
            a0 = 1 + (resi[int(outlet_ind[i, 0]), 0]) / (
                resi[int(outlet_ind[i, 0]), 1] + 1e-6
            )
            a1 = resi[int(outlet_ind[i, 0]), 0] * resi[int(outlet_ind[i, 0]), 2]
            a2 = 1 / (resi[int(outlet_ind[i, 0]), 1] + 1e-6)
            # a2 = 1
            a3 = resi[int(outlet_ind[i, 0]), 2]
            # print(f"printing out a terms {a0 + a1 / dt, a1, a2, a3}")
            M3[NJ + NT + NQ + i, int(outlet_ind[i, 0])] = a0 + 1 * a1 / dt
            # print(M3)
            M3[NJ + NT + NQ + i, NV + NJ + NT + NQ + i] = -a2 - 1 * a3 / dt
            b[NV + NJ + NT + NQ + i, 0] = (
                1 * a1 / dt * X_1[int(outlet_ind[i, 0]), 0]
                - 1 * a3 / dt * X_1[NV + NJ + NT + NQ + i, 0]
            )
            M2[int(outlet_ind[i, 0]), NJ + NT + NQ + i] = -1

        M3[NJ + NT + NQ + NO, 0] = 1
        M2[0, -1] = 1
        b[NV + NJ + NT + NQ + NO, 0] = Qin[c, 0]
        M12 = np.concatenate((M1, M2), axis=1)
        M = np.concatenate((M12, M3), axis=0)
        cond = np.linalg.cond(M)
        X = np.linalg.solve(M, b)
        if np.any(np.isnan(X)):
            break

        Pin[c, 0] = X[6, 0]
        Pout1[c, 0] = X[3, 0]
        Pout2[c, 0] = X[5, 0]
        Qout1[c, 0] = X[1, 0]
        Qout2[c, 0] = X[2, 0]

        q0 = X[0, 0]
        q1 = X[1, 0]
        q2 = X[2, 0]
        p01 = X[3, 0]
        p1o = X[4, 0]
        p2o = X[5, 0]
        pin = X[6, 0]

        if (
            np.linalg.norm(X[0:NV, 0] - X0[0:NV, 0], np.inf) > 1e-5
            and np.linalg.norm(X[NV:, 0] - X0[NV:, 0], np.inf) > 1e-2
            and k < 10
        ):
            X0 = X
        else:
            s = 0
    X_2 = X_1
    X_1 = X

    if np.mod(c, T / dt / 100) == 0:
        tt[c4, 0] = dt * c
        Q0[c4, 0] = q0
        Q1[c4, 0] = q1
        Q2[c4, 0] = q2
        P1[c4, 0] = 1 * p1o
        P2[c4, 0] = 1 * p2o
        PJ01[c4, 0] = p01
        P0[c4, 0] = 1 * pin
        c4 = c4 + 1

print(Pin.shape, Pout1.shape, Pout2.shape, Qout1.shape, Qout2.shape)
tracked_data = np.stack(
    (Pin[:, 0], Pout1[:, 0], Pout2[:, 0], Qout1[:, 0], Qout2[:, 0]), axis=1
)

np.savetxt("tracked_data_old.txt", tracked_data)
