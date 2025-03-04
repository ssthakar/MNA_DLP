import numpy as np
import scipy


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
